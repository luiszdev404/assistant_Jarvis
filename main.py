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
import re
import threading
import traceback
from datetime import datetime

import sounddevice as sd
from google import genai
from google.genai import types

from core.settings import (
    get_api_key,
    load_system_prompt,
    LIVE_MODEL,
    CHANNELS,
    SEND_SAMPLE_RATE,
    RECEIVE_SAMPLE_RATE,
    CHUNK_SIZE,
    VOICE_NAME,
)
from core.skill_registry import SkillRegistry
from memory.memory_manager import load_memory, update_memory, format_memory_for_prompt
from ui import JarvisUI


# ── Transcript cleaner ─────────────────────────────────────────────────────────
_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

def _clean_transcript(text: str) -> str:
    """Remove Gemini artifact tokens and control characters."""
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()


# ── Save-memory tool declaration ───────────────────────────────────────────────
_SAVE_MEMORY_DECLARATION = {
    "name": "save_memory",
    "description": (
        "Save an important personal fact about the user to long-term memory. "
        "Call silently whenever the user reveals: name, age, city, job, preferences, "
        "hobbies, projects, or future plans. Do NOT announce that you are saving."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "category": {
                "type": "STRING",
                "description": (
                    "identity | preferences | projects | relationships | wishes | notes"
                ),
            },
            "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, city)"},
            "value": {"type": "STRING", "description": "Concise value in English"},
        },
        "required": ["category", "key", "value"],
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
        self.registry        = SkillRegistry(api_key=self.api_key)
        self.session         = None
        self.audio_in_queue  = None
        self.out_queue       = None
        self._loop           = None
        self._is_speaking    = False
        self._speaking_lock  = threading.Lock()

        self.ui.on_text_command = self._on_text_command

    # ── UI callbacks ───────────────────────────────────────────────────────────

    def _on_text_command(self, text: str) -> None:
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True,
            ),
            self._loop,
        )

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
        all_declarations   = skill_declarations + [_SAVE_MEMORY_DECLARATION, _SHUTDOWN_DECLARATION]

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
                )
            ),
        )

    # ── Tool execution ─────────────────────────────────────────────────────────

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        self.ui.write_log(f"SYS: {name}  {args}")
        self.ui.set_state("THINKING")

        # Built-in: save_memory (silent, fast)
        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                self.ui.write_log(f"SYS: saved {category}/{key}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True},
            )

        # Built-in: shutdown
        if name == "shutdown_jarvis":
            self.ui.write_log("SYS: Shutdown requested.")
            self.speak("Goodbye, sir. Shutting down.")
            def _exit():
                import time, os
                time.sleep(1.5)
                os._exit(0)
            threading.Thread(target=_exit, daemon=True).start()
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "shutting_down"},
            )

        # Skill dispatch
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.registry.execute(name, args)
        )

        if not self.ui.muted:
            self.ui.set_state("LISTENING")


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
        loop = asyncio.get_event_loop()

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
            with self._speaking_lock:
                jarvis_speaking = self._is_speaking
            if not jarvis_speaking and not self.ui.muted:
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

    async def _receive_audio(self) -> None:
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    if response.data:
                        self.audio_in_queue.put_nowait(response.data)

                    if response.server_content:
                        sc = response.server_content

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
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)

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

        except Exception as e:
            self.ui.write_log(f"ERR: receive — {e}")
            traceback.print_exc()
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
        client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1beta"},
        )

        while True:
            try:
                self.ui.write_log("SYS: 🔌 Connecting to Gemini Live...")
                self.ui.set_state("THINKING")
                config = self._build_config()

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session        = session
                    self._loop          = asyncio.get_event_loop()
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
    ui = JarvisUI()
    ui.start_input_loop()

    jarvis = Jarvis(ui)

    try:
        asyncio.run(jarvis.run())
    except KeyboardInterrupt:
        ui.write_log("SYS: Interrupted by user.")
        ui.shutdown()
    except Exception as e:
        ui.shutdown()
        import time
        time.sleep(0.1)  # Let the curses thread clean up the terminal
        traceback.print_exc()


if __name__ == "__main__":
    main()
