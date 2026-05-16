"""
main.py — Jarvis2 entry point.

Architecture:
  - Gemini Live API for real-time voice conversation (bidirectional audio)
  - SkillRegistry handles all tool dispatch (no if/elif chains)
  - JarvisUI renders the terminal interface

Flow:
  1. Load config and build SkillRegistry
  2. Connect to Gemini Live session
  3. Stream audio in/out concurrently
  4. On tool_call → registry.execute(name, params) → send_tool_response
  5. On disconnect → reconnect automatically
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime

import sounddevice as sd
from google import genai
from google.genai import types

from core.settings import (
    get_api_key,
    get_skill_api_key,
    load_system_prompt,
    LIVE_MODEL,
    CHANNELS,
    SEND_SAMPLE_RATE,
    RECEIVE_SAMPLE_RATE,
    CHUNK_SIZE,
    VOICE_NAME,
)
from core.skill_registry import SkillRegistry

_STOP_WORDS = frozenset({"para", "detente", "cancela", "basta", "stop", "olvídalo", "olvida"})
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    forget_memory, recall_memory, expire_memory,
)
from skills.tech_researcher import _READ_PREFIX
from ui import JarvisUI


# ── Transcript cleaner ─────────────────────────────────────────────────────────
_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

def _clean_transcript(text: str) -> str:
    """Remove Gemini artifact tokens and control characters."""
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()


# ── Memory tool declarations ───────────────────────────────────────────────────
_SAVE_MEMORY_DECLARATION = {
    "name": "save_memory",
    "description": (
        "Save an important fact about the user to long-term memory. "
        "Call silently whenever the user reveals personal info, preferences, projects, "
        "routines, tech stack, or plans. Do NOT announce that you are saving. "
        "Use expires_in_days for temporary context (e.g. 'this week I am working on X' → 7 days)."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "category": {
                "type": "STRING",
                "description": (
                    "identity (name, age, location, occupation) | "
                    "preferences (music, news, food, language, communication style) | "
                    "proyectos (active projects with status) | "
                    "stack (programming languages, tools, frameworks) | "
                    "rutinas (work hours, habits, schedule) | "
                    "relaciones (important people mentioned) | "
                    "contexto (current week task — use expires_in_days=7) | "
                    "notas (anything else)"
                ),
            },
            "key":   {"type": "STRING", "description": "Short snake_case key (e.g. nombre, ciudad, ide_favorito)"},
            "value": {"type": "STRING", "description": "Concise value in Spanish"},
            "expires_in_days": {
                "type": "INTEGER",
                "description": "Optional TTL in days. Use for temporary context like current tasks or weekly plans.",
            },
        },
        "required": ["category", "key", "value"],
    },
}

_FORGET_MEMORY_DECLARATION = {
    "name": "forget_memory",
    "description": (
        "Delete a specific memory entry or an entire category. "
        "Call when the user says 'olvida que...', 'borra que...', or corrects outdated info."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "category": {"type": "STRING", "description": "The category to target"},
            "key":      {"type": "STRING", "description": "The specific key to delete. Omit to delete the entire category."},
        },
        "required": ["category"],
    },
}

_RECALL_MEMORY_DECLARATION = {
    "name": "recall_memory",
    "description": (
        "Query long-term memory. Use when the user asks what you remember, "
        "or when you need to retrieve a specific fact before answering. "
        "Provide either a keyword query or a category name."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "query":    {"type": "STRING", "description": "Keyword to search across all memory"},
            "category": {"type": "STRING", "description": "Return all entries in this category"},
        },
        "required": [],
    },
}

_SHUTDOWN_DECLARATION = {
    "name": "shutdown_jarvis",
    "description": (
        "Shuts down the assistant. Call when the user says goodbye, "
        "exit, close, or stop Jarvis in any language."
    ),
    "parameters": {"type": "OBJECT", "properties": {}},
}


# ── Main Jarvis class ──────────────────────────────────────────────────────────

class Jarvis:
    """Real-time voice assistant powered by Gemini Live."""

    def __init__(self, ui: JarvisUI) -> None:
        self.ui              = ui
        self.api_key         = get_api_key()
        self.registry        = SkillRegistry(
            api_key=self.api_key,
            skill_api_key=get_skill_api_key(),
        )
        self.session         = None
        self.audio_in_queue  = None
        self.out_queue       = None
        self._loop           = None
        self._is_speaking    = False
        self._speaking_lock  = threading.Lock()

        self.ui.on_text_command = self.speak
        self._pending_read: str | None = None

        # Wire completion notifications from background skills to the voice session
        self.registry.register_notify_fn(("tech_researcher", "news_journalist"), self.speak)

        # Purge expired memory entries on every startup
        removed = expire_memory()
        if removed:
            self.ui.write_log(f"SYS: Expired {removed} memory entries.")

    # ── UI callbacks ───────────────────────────────────────────────────────────

    def set_speaking(self, value: bool) -> None:
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def speak(self, text: str) -> None:
        """Send a text message to the Gemini session (causes audio response)."""
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True,
            ),
            self._loop,
        )

    # ── Session configuration ──────────────────────────────────────────────────

    def _build_config(self) -> types.LiveConnectConfig:
        memory  = load_memory()
        mem_str = format_memory_for_prompt(memory)
        prompt  = load_system_prompt()

        now      = datetime.now()
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Now: {now.strftime('%A, %B %d, %Y — %I:%M %p')}\n\n"
        )

        parts = [time_ctx]
        if mem_str:
            parts.append(mem_str)
        parts.append(prompt)
        system_instruction = "\n".join(parts)

        # Combine skill declarations + built-in declarations
        skill_declarations = self.registry.get_tool_declarations()
        all_declarations   = skill_declarations + [
            _SAVE_MEMORY_DECLARATION,
            _FORGET_MEMORY_DECLARATION,
            _RECALL_MEMORY_DECLARATION,
            _SHUTDOWN_DECLARATION,
        ]

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction=system_instruction,
            tools=[{"function_declarations": all_declarations}],
            session_resumption=types.SessionResumptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=VOICE_NAME
                    )
                ),
            ),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    # ── Tool execution ─────────────────────────────────────────────────────────

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        self.ui.write_log(f"SYS: {name}  {args}")
        self.ui.set_state("THINKING")

        # Built-in: save_memory (silent, fast)
        if name == "save_memory":
            category       = args.get("category", "notas")
            key            = args.get("key", "")
            value          = args.get("value", "")
            expires_in_days = args.get("expires_in_days")
            if key and value:
                update_memory(
                    {category: {key: {"value": value}}},
                    expires_in_days=int(expires_in_days) if expires_in_days else None,
                )
                self.ui.write_log(f"SYS: memory saved {category}/{key}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True},
            )

        # Built-in: forget_memory
        if name == "forget_memory":
            category = args.get("category", "")
            key      = args.get("key")
            result   = forget_memory(category, key or None)
            self.ui.write_log(f"SYS: forget_memory {category}/{key} → {result}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": result},
            )

        # Built-in: recall_memory
        if name == "recall_memory":
            query    = args.get("query")
            category = args.get("category")
            result   = recall_memory(query=query, category=category)
            self.ui.write_log(f"SYS: recall_memory query={query} cat={category}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": result},
            )

        # Built-in: shutdown
        if name == "shutdown_jarvis":
            self.ui.write_log("SYS: Shutdown requested.")
            self.speak("Goodbye, sir. Shutting down.")
            def _exit():
                time.sleep(1.5)
                os._exit(0)
            threading.Thread(target=_exit, daemon=True).start()
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "shutting_down"},
            )

        # Skill dispatch
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: self.registry.execute(name, args)
        )

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        # If the skill returned content to be read aloud, intercept it.
        # We store it and send it as a client turn AFTER the tool response so
        # Gemini reads it as a new instruction rather than just acknowledging
        # the function response.
        if isinstance(result, str) and result.startswith(_READ_PREFIX):
            self._pending_read = result[len(_READ_PREFIX):]
            result = "Sending the document for reading."

        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result},
        )

    # ── Audio tasks ────────────────────────────────────────────────────────────

    async def _send_realtime(self) -> None:
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(media=msg)

    async def _listen_audio(self) -> None:
        loop = asyncio.get_running_loop()

        def _safe_enqueue(item: dict) -> None:
            """Put audio frame in queue, silently drop if it gets too large."""
            # If queue is getting too large (connection stalled), drop oldest
            if self.out_queue.qsize() > 10:
                try:
                    self.out_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self.out_queue.put_nowait(item)

        def callback(indata, frames, time_info, status):
            if not self.ui.muted:
                data = indata.tobytes()
                loop.call_soon_threadsafe(
                    _safe_enqueue,
                    {"data": data, "mime_type": "audio/pcm"},
                )

        with sd.InputStream(
            samplerate=SEND_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            callback=callback,
        ):
            self.ui.write_log("SYS: Microphone active.")
            while True:
                await asyncio.sleep(0.1)

    def _drain_audio(self) -> None:
        """Drain buffered audio so playback stops immediately on interruption."""
        drained = 0
        while not self.audio_in_queue.empty():
            try:
                self.audio_in_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            self.ui.write_log(f"SYS: Drained {drained} audio chunks.")
        self.set_speaking(False)

    async def _receive_audio(self) -> None:
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    if response.data:
                        self.audio_in_queue.put_nowait(response.data)

                    if response.server_content:
                        sc = response.server_content

                        # Gemini native interruption signal — drain buffered audio immediately
                        if getattr(sc, "interrupted", False):
                            self._drain_audio()

                        if sc.output_transcription and sc.output_transcription.text:
                            self.set_speaking(True)
                            txt = _clean_transcript(sc.output_transcription.text)
                            if txt:
                                out_buf.append(txt)

                        if sc.model_turn and sc.model_turn.parts:
                            for part in sc.model_turn.parts:
                                if part.text:
                                    txt = _clean_transcript(part.text)
                                    if txt and txt not in out_buf:
                                        out_buf.append(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            # User spoke while Jarvis was talking — drain buffered audio
                            if self._is_speaking:
                                self._drain_audio()
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)
                                # Cancel background tasks if user said a stop word
                                words = set(txt.lower().split())
                                if words & _STOP_WORDS:
                                    self.registry.cancel_background_tasks()

                        if sc.turn_complete:
                            self.set_speaking(False)

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            if full_out:
                                self.ui.write_log(f"Jarvis: {full_out}")
                            out_buf = []

                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(function_responses=fn_responses)

                        # Send any pending read content as a new client turn so
                        # Gemini reads it aloud instead of just acknowledging it.
                        if self._pending_read:
                            await self.session.send_client_content(
                                turns={"parts": [{"text": self._pending_read}]},
                                turn_complete=True,
                            )
                            self._pending_read = None

        except Exception as e:
            self.ui.write_log(f"ERR: receive — {e}")
            raise

    async def _play_audio(self) -> None:
        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        )
        stream.start()
        try:
            while True:
                chunk = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, chunk)
        except Exception as e:
            self.ui.write_log(f"ERR: play — {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    # ── Main run loop ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        while True:
            try:
                # Re-read key each cycle so changes to api_keys.json take effect
                # without restarting Jarvis.
                self.api_key = get_api_key()
                client = genai.Client(
                    api_key=self.api_key,
                    http_options={"api_version": "v1beta"},
                )

                self.ui.write_log("SYS: Connecting to Gemini Live...")
                self.ui.set_state("THINKING")
                config = self._build_config()

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session        = session
                    self._loop          = asyncio.get_running_loop()
                    # Fresh queues — drain any stale audio from previous cycle
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue      = asyncio.Queue(maxsize=10)

                    self.ui.set_state("LISTENING")
                    self.ui.write_log("SYS: JARVIS 2.0 online. Ready, sir.")

                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())

            except Exception as e:
                self.ui.write_log(f"ERR: {e}")
                traceback.print_exc()

            self.set_speaking(False)
            self.ui.set_state("THINKING")
            self.ui.write_log("SYS: Reconnecting in 3s...")
            await asyncio.sleep(3)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Redirect stdout/stderr to a log file so that print() calls inside
    # skills (and any library debug output) never corrupt the curses display.
    _log_path = os.path.expanduser("~/.jarvis2.log")
    _log_file = open(_log_path, "a", buffering=1)
    sys.stdout = _log_file
    sys.stderr = _log_file

    ui = JarvisUI()
    ui.start_input_loop()

    jarvis = Jarvis(ui)

    try:
        asyncio.run(jarvis.run())
    except KeyboardInterrupt:
        ui.write_log("SYS: Interrupted by user.")
        ui.shutdown()
    except Exception as e:
        ui.write_log(f"ERR: {e}")
        ui.shutdown()
        time.sleep(0.3)


if __name__ == "__main__":
    main()
