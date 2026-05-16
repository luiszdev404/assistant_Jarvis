"""
ui.py — Jarvis2 terminal UI for Linux GNOME (curses edition).

Layout (curses):
  ┌─────────────────────────────────────────────┐
  │  ╔═══════════════════════════════════════╗  │  ← banner (5 lines, drawn once)
  │  ║          J.A.R.V.I.S  —  Linux        ║  │
  │  ╚═══════════════════════════════════════╝  │
  ├── [STATUS] ──────────────────────────────── │  ← status bar (1 line, always row 5)
  │                                             │
  │   (scrollable conversation log)             │  ← log area (dynamic height)
  │                                             │
  ├─────────────────────────────────────────────│  ← separator
  │  ❯  _                                       │  ← STATIC input box (last 3 rows)
  └─────────────────────────────────────────────┘

Thread safety: ALL curses calls happen exclusively in the curses thread.
Other threads post drawing requests to _draw_q (a queue.SimpleQueue); the
curses thread drains it between getch() calls. This avoids ncurses corruption.

Public API (same as original JarvisUI):
    set_state(state)
    write_log(message)
    toggle_mute()
    start_input_loop()
    shutdown()
    muted : bool
    on_text_command : Callable[[str], None]
"""
from __future__ import annotations

import curses
import queue
import threading
import time
from datetime import datetime
from typing import Callable


# ── State / label tables ───────────────────────────────────────────────────────
_STATES: dict[str, tuple[str, int]] = {
    "LISTENING": ("◉  LISTENING",   1),
    "SPEAKING":  ("▶  SPEAKING",    2),
    "THINKING":  ("◌  THINKING",    3),
    "MUTED":     ("✕  MUTED",       4),
    "OFFLINE":   ("○  OFFLINE",     5),
}

_LOG_TAGS: dict[str, tuple[str, int]] = {
    "You":    ("You   ", 1),
    "Jarvis": ("Jarvis", 2),
    "SYS":    ("System", 5),
    "ERR":    ("Error ", 4),
}

_BANNER_TITLE = "J · A · R · V · I · S"
_BANNER_H     = 3        # blank + title + separator
_STATUS_ROW   = _BANNER_H          # row 3
_DIAG_ROW     = _STATUS_ROW + 1   # row 4  ← diagnostic bar
_LOG_START    = _DIAG_ROW + 1     # row 5

_INPUT_ROWS = 3   # separator + prompt + hint

# Draw-queue message types
_DQ_LOG    = "log"
_DQ_STATUS = "status"
_DQ_DIAG   = "diag"
_DQ_FULL   = "full"

# Diagnostic bar levels → (prefix, color_pair)
_DIAG_LEVELS: dict[str, tuple[str, int]] = {
    "ok":      ("✓", 1),   # green
    "info":    ("◌", 5),   # gray
    "warn":    ("⚠", 3),   # yellow
    "error":   ("✗", 4),   # red
    "busy":    ("◎", 2),   # cyan
}

# Patterns that auto-populate the diagnostic bar from write_log messages.
# Columns: (substring_to_match, level, label, ttl_seconds)
# ttl=0 → sticky (stays until overridden by another pattern)
_DIAG_PATTERNS: list[tuple[str, str, str, int]] = [
    # ── Quota / rate-limit (sticky or long) ────────────────────────────────────
    ("RESOURCE_EXHAUSTED",          "error", "⚠ Cuota Gemini agotada — usando DDG",    0),
    ("quota",                       "error", "⚠ Cuota Gemini agotada — usando DDG",    0),
    ("429",                         "warn",  "Rate limit Gemini — reintentando...",    90),
    ("403 Ratelimit",               "warn",  "Rate limit DDG — reintentando...",       30),
    ("Ratelimit",                   "warn",  "Rate limit — reintentando...",           30),
    ("All backends failed",         "error", "Sin backends disponibles",               0),
    ("Todas las fuentes fallaron",  "error", "Sin fuentes disponibles",                0),
    # ── Fallback activo ─────────────────────────────────────────────────────────
    ("raw DDG results",             "warn",  "Gemini sin cuota — resultados DDG crudos", 20),
    ("DDG news agotado",            "warn",  "DDG agotado — buscando alternativa",     15),
    # ── Éxito (limpia avisos de cuota) ─────────────────────────────────────────
    ("Gemini OK",                   "ok",    "Gemini disponible",                       6),
    ("BACKGROUND TASK COMPLETE",    "ok",    "Tarea completada",                        8),
    # ── Tareas en curso ─────────────────────────────────────────────────────────
    ("BUSCANDO EN SEGUNDO PLANO",   "busy",  "Buscando noticias...",                    0),
    ("Investigando",                "busy",  "Investigando tema...",                    0),
    ("BACKGROUND TASK FAILED",      "error", "Error en tarea de fondo",                15),
    # ── Sistema ─────────────────────────────────────────────────────────────────
    ("Connecting to Gemini",        "busy",  "Conectando con Gemini...",                0),
    ("JARVIS 2.0 online",           "ok",    "Jarvis conectado",                        8),
    ("Reconnecting",                "warn",  "Reconectando...",                        15),
    ("Internal error encountered",  "error", "Error interno del servidor",             15),
    ("Drained",                     "info",  "Audio interrumpido",                      5),
    ("Cancelado",                   "info",  "Tarea cancelada",                         5),
    ("Expired",                     "info",  "Memoria: entradas expiradas",             5),
]


class JarvisUI:
    """
    Curses-based terminal UI for Jarvis2.

    All ncurses operations run exclusively in the curses thread.
    Other threads enqueue draw requests via _draw_q.
    """

    def __init__(self) -> None:
        self._state      = "OFFLINE"
        self._last_state = ""
        self.muted       = False
        self.on_text_command: Callable[[str], None] | None = None

        self._log: list[tuple[str, str, str]] = []
        self._log_lock = threading.Lock()

        self._pulse_stop  = threading.Event()
        self._pulse_label = ""
        self._pulse_pair  = 5
        self._pulse_lock  = threading.Lock()

        self._scr: curses.window | None = None
        self._rows = 24
        self._cols = 80

        self._input_buf  = ""
        self._cursor_pos = 0
        self._scroll_offset = 0

        # Diagnostic bar state
        self._diag_text  = ""
        self._diag_pair  = 5
        self._diag_timer: threading.Timer | None = None
        self._diag_lock  = threading.Lock()

        # Thread-safe queue: other threads post draw requests here;
        # the curses thread processes them between getch() calls.
        self._draw_q: queue.SimpleQueue = queue.SimpleQueue()

        self._ready = threading.Event()
        self._alive = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def start_input_loop(self) -> None:
        t = threading.Thread(target=self._run_curses, daemon=True)
        t.start()
        self._ready.wait(timeout=5)

    def set_state(self, state: str) -> None:
        with self._pulse_lock:
            if self.muted and state == "LISTENING":
                state = "MUTED"
            if state == self._last_state:
                return
            self._last_state  = state
            self._state       = state
            label, pair       = _STATES.get(state, ("◌  UNKNOWN", 5))
            self._pulse_label = label
            self._pulse_pair  = pair
        self._draw_q.put((_DQ_STATUS, label, pair))
        self._restart_pulse(label, pair)

    def set_skill_status(self, msg: str, level: str = "info", ttl: int = 8) -> None:
        """Update the diagnostic bar. ttl=0 keeps it until next update."""
        prefix, pair = _DIAG_LEVELS.get(level, ("◌", 5))
        text = f"  {prefix}  {msg}"
        with self._diag_lock:
            if self._diag_timer:
                self._diag_timer.cancel()
            self._diag_text = text
            self._diag_pair = pair
            if ttl > 0:
                self._diag_timer = threading.Timer(ttl, self._clear_diag)
                self._diag_timer.daemon = True
                self._diag_timer.start()
        self._draw_q.put((_DQ_DIAG, text, pair))

    def _clear_diag(self) -> None:
        with self._diag_lock:
            self._diag_text = ""
            self._diag_pair = 5
        self._draw_q.put((_DQ_DIAG, "", 5))

    def write_log(self, message: str) -> None:
        tag  = "SYS"
        body = message
        for t in _LOG_TAGS:
            if message.startswith(f"{t}:"):
                tag  = t
                body = message[len(t) + 1:].strip()
                break
        ts = datetime.now().strftime("%H:%M:%S")
        with self._log_lock:
            self._log.append((tag, ts, body))
            self._scroll_offset = 0
        self._draw_q.put((_DQ_LOG,))
        # Auto-detect patterns and update diagnostic bar
        self._auto_diag(message)

    def _auto_diag(self, message: str) -> None:
        """Scan message for known patterns and update the diagnostic bar."""
        for pattern, level, label, ttl in _DIAG_PATTERNS:
            if pattern in message:
                self.set_skill_status(label, level, ttl=ttl)
                return

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        if self.muted:
            self._last_state = ""
            self.set_state("MUTED")
            self.write_log("SYS: Microphone muted.")
        else:
            self._last_state = ""
            self.set_state("LISTENING")
            self.write_log("SYS: Microphone active.")

    def shutdown(self) -> None:
        self._pulse_stop.set()
        self._alive      = False
        self._last_state = ""
        self.set_state("OFFLINE")

    # ── Curses initialisation ──────────────────────────────────────────────────

    def _run_curses(self) -> None:
        curses.wrapper(self._curses_main)

    def _curses_main(self, scr: curses.window) -> None:
        self._scr = scr
        curses.curs_set(0)
        curses.noecho()
        curses.raw()
        scr.keypad(True)
        scr.timeout(100)   # non-blocking getch (100 ms)

        self._init_colors()
        self._rows, self._cols = scr.getmaxyx()
        self._full_redraw()
        self._ready.set()

        while self._alive:
            try:
                ch = scr.getch()
            except Exception:
                time.sleep(0.05)
                continue

            # Detect terminal resize
            new_rows, new_cols = scr.getmaxyx()
            if new_rows != self._rows or new_cols != self._cols:
                self._rows, self._cols = new_rows, new_cols
                self._full_redraw()
                continue

            # ── Drain draw queue (all curses I/O lives here) ────────────────
            needs_refresh = False
            while True:
                try:
                    req = self._draw_q.get_nowait()
                except queue.Empty:
                    break
                if req[0] == _DQ_LOG:
                    self._draw_log_region()
                    self._draw_input_box()
                    needs_refresh = True
                elif req[0] == _DQ_STATUS:
                    self._draw_status_line(req[1], req[2])
                    needs_refresh = True
                elif req[0] == _DQ_DIAG:
                    self._draw_diag_bar(req[1], req[2])
                    needs_refresh = True
                elif req[0] == _DQ_FULL:
                    self._full_redraw()
                    needs_refresh = False   # _full_redraw already refreshes

            if needs_refresh:
                scr.refresh()

            if ch == curses.ERR:
                continue

            self._handle_key(ch)

    def _init_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN,  -1)
        curses.init_pair(2, curses.COLOR_CYAN,   -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED,    -1)
        curses.init_pair(5, 8,                   -1)
        curses.init_pair(6, curses.COLOR_CYAN,   -1)
        curses.init_pair(7, curses.COLOR_WHITE,  -1)

    # ── Full redraw (curses thread only) ───────────────────────────────────────

    def _full_redraw(self) -> None:
        if not self._scr:
            return
        self._scr.erase()
        self._draw_banner()
        label, pair = _STATES.get(self._state, ("○  OFFLINE", 5))
        self._draw_status_line(label, pair)
        self._draw_diag_bar(self._diag_text, self._diag_pair)
        self._draw_log_region()
        self._draw_input_box()
        self._scr.refresh()

    # ── Banner ─────────────────────────────────────────────────────────────────

    def _draw_banner(self) -> None:
        col = max(0, (self._cols - len(_BANNER_TITLE)) // 2)
        sep = "─" * len(_BANNER_TITLE)
        # row 0: blank
        self._safe_addstr(0, 0, " " * (self._cols - 1), 0)
        # row 1: title centered
        self._safe_addstr(1, col, _BANNER_TITLE,
                          curses.color_pair(6) | curses.A_BOLD)
        # row 2: thin separator centered
        self._safe_addstr(2, col, sep, curses.color_pair(5))

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _draw_status_line(self, label: str, pair_idx: int) -> None:
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"  {ts}  {label}".ljust(self._cols - 1)
        self._safe_addstr(_STATUS_ROW, 0, line[:self._cols - 1],
                          curses.color_pair(pair_idx) | curses.A_BOLD)

    # ── Diagnostic bar ─────────────────────────────────────────────────────────

    def _draw_diag_bar(self, text: str, pair_idx: int) -> None:
        line = (text or "").ljust(self._cols - 1)
        self._safe_addstr(_DIAG_ROW, 0, line[:self._cols - 1],
                          curses.color_pair(pair_idx))

    # ── Log region ─────────────────────────────────────────────────────────────

    def _log_area_height(self) -> int:
        return max(1, self._rows - _LOG_START - _INPUT_ROWS)

    def _draw_log_region(self) -> None:
        height = self._log_area_height()
        width  = self._cols

        with self._log_lock:
            entries = list(self._log)
            offset  = self._scroll_offset

        display_lines: list[tuple[int, str, str]] = []
        for tag, ts, body in entries:
            pfx_text, pair_idx = _LOG_TAGS.get(tag, ("      ", 7))
            prefix_str = f"  {ts}  {pfx_text}  │  "
            max_body   = max(1, width - len(prefix_str) - 1)
            words      = body.split()
            line_acc   = ""
            first      = True
            if not words:
                display_lines.append((pair_idx, prefix_str, ""))
                continue
            for word in words:
                while len(word) > max_body:
                    chunk = word[:max_body]
                    if line_acc:
                        display_lines.append((pair_idx,
                                              prefix_str if first else " " * len(prefix_str),
                                              line_acc))
                        first    = False
                        line_acc = ""
                    display_lines.append((pair_idx,
                                          prefix_str if first else " " * len(prefix_str),
                                          chunk))
                    first = False
                    word  = word[max_body:]
                if not word:
                    continue
                if len(line_acc) + (1 if line_acc else 0) + len(word) <= max_body:
                    line_acc += (" " if line_acc else "") + word
                else:
                    if line_acc:
                        display_lines.append((pair_idx,
                                              prefix_str if first else " " * len(prefix_str),
                                              line_acc))
                        first    = False
                    line_acc = word
            if line_acc:
                display_lines.append((pair_idx,
                                      prefix_str if first else " " * len(prefix_str),
                                      line_acc))

        total   = len(display_lines)
        start   = max(0, total - height - offset)
        end     = max(0, total - offset)
        visible = display_lines[start:end]
        blank_rows = height - len(visible)

        for r in range(height):
            row = _LOG_START + r
            if r < blank_rows:
                self._safe_addstr(row, 0, " " * (width - 1), curses.color_pair(7))
            else:
                idx = r - blank_rows
                pair_idx, prefix_str, body = visible[idx]
                full_line = (prefix_str + body).ljust(width - 1)
                self._safe_addstr(row, 0, full_line[:width - 1],
                                  curses.color_pair(7) | curses.A_DIM)
                if len(prefix_str) < width:
                    self._safe_addstr(row, 0, prefix_str[:width - 1],
                                      curses.color_pair(pair_idx) | curses.A_BOLD)

    # ── Input box (static, pinned to bottom) ──────────────────────────────────

    def _draw_input_box(self) -> None:
        rows, width = self._scr.getmaxyx()

        sep_row    = rows - _INPUT_ROWS
        prompt_row = rows - _INPUT_ROWS + 1
        hint_row   = rows - 1

        sep = "─" * (width - 1)
        self._safe_addstr(sep_row, 0, sep, curses.color_pair(5))

        prompt   = "  ❯  "
        prompt_p = curses.color_pair(2) | curses.A_BOLD
        self._safe_addstr(prompt_row, 0, prompt, prompt_p)

        buf_start = len(prompt)
        buf_area  = width - buf_start - 1
        buf_text  = self._input_buf
        if len(buf_text) > buf_area:
            buf_text = buf_text[-buf_area:]
        padded = buf_text.ljust(buf_area)
        self._safe_addstr(prompt_row, buf_start, padded[:buf_area], curses.color_pair(7))

        cursor_col  = buf_start + min(self._cursor_pos, buf_area - 1)
        char_under  = padded[min(self._cursor_pos, len(padded) - 1)] if padded else " "
        self._safe_addstr(prompt_row, cursor_col, char_under,
                          curses.color_pair(3) | curses.A_REVERSE)

        hint = "  [Enter] send  [m] mute  [↑/↓] scroll  [Ctrl+C] exit"
        self._safe_addstr(hint_row, 0, hint[:width - 1], curses.color_pair(5))

    # ── Key handling (curses thread — may call draw directly) ─────────────────

    def _handle_key(self, ch: int) -> None:
        scr = self._scr
        if ch in (curses.KEY_ENTER, 10, 13):
            self._submit_input()
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if self._cursor_pos > 0:
                pos = self._cursor_pos
                self._input_buf  = self._input_buf[:pos - 1] + self._input_buf[pos:]
                self._cursor_pos -= 1
            self._draw_input_box()
            scr.refresh()
        elif ch == curses.KEY_LEFT:
            if self._cursor_pos > 0:
                self._cursor_pos -= 1
            self._draw_input_box()
            scr.refresh()
        elif ch == curses.KEY_RIGHT:
            if self._cursor_pos < len(self._input_buf):
                self._cursor_pos += 1
            self._draw_input_box()
            scr.refresh()
        elif ch == curses.KEY_UP:
            self._scroll_offset += 1
            self._draw_log_region()
            self._draw_input_box()
            scr.refresh()
        elif ch == curses.KEY_DOWN:
            self._scroll_offset = max(0, self._scroll_offset - 1)
            self._draw_log_region()
            self._draw_input_box()
            scr.refresh()
        elif ch == curses.KEY_HOME:
            self._cursor_pos = 0
            self._draw_input_box()
            scr.refresh()
        elif ch == curses.KEY_END:
            self._cursor_pos = len(self._input_buf)
            self._draw_input_box()
            scr.refresh()
        elif ch == curses.KEY_DC:
            pos = self._cursor_pos
            if pos < len(self._input_buf):
                self._input_buf = self._input_buf[:pos] + self._input_buf[pos + 1:]
            self._draw_input_box()
            scr.refresh()
        elif ch == 3:   # Ctrl+C
            self.shutdown()
            raise KeyboardInterrupt
        elif 32 <= ch < 256:
            pos = self._cursor_pos
            self._input_buf  = self._input_buf[:pos] + chr(ch) + self._input_buf[pos:]
            self._cursor_pos += 1
            self._draw_input_box()
            scr.refresh()

    def _submit_input(self) -> None:
        cmd = self._input_buf.strip()
        self._input_buf  = ""
        self._cursor_pos = 0
        self._draw_input_box()
        self._scr.refresh()

        if not cmd:
            return
        if cmd.lower() == "m":
            self.toggle_mute()
            return

        self.write_log(f"You: {cmd}")
        if self.on_text_command:
            self.on_text_command(cmd)

    # ── Pulse animation ────────────────────────────────────────────────────────

    def _restart_pulse(self, label: str, pair_idx: int) -> None:
        self._pulse_stop.set()
        if self._state not in ("THINKING",):
            return
        self._pulse_stop = threading.Event()
        t = threading.Thread(
            target=self._pulse_loop,
            args=(label, pair_idx, self._pulse_stop),
            daemon=True,
        )
        t.start()

    def _pulse_loop(self, label: str, pair_idx: int, stop: threading.Event) -> None:
        dots = ["·  ", "·· ", "···", " ··", "  ·", "   "]
        i    = 0
        while not stop.is_set():
            stop.wait(0.4)
            if stop.is_set():
                break
            with self._pulse_lock:
                if self._pulse_label != label:
                    break
            suffix = "  " + dots[i % len(dots)]
            self._draw_q.put((_DQ_STATUS, label + suffix, pair_idx))
            i += 1

    # ── Curses safe helper ─────────────────────────────────────────────────────

    def _safe_addstr(self, row: int, col: int, text: str, attr: int = 0) -> None:
        scr = self._scr
        if scr is None:
            return
        try:
            max_r, max_c = scr.getmaxyx()
            if row < 0 or row >= max_r or col < 0 or col >= max_c:
                return
            avail = max_c - col - 1
            if avail <= 0:
                return
            scr.addstr(row, col, text[:avail], attr)
        except curses.error:
            pass
