"""
ui.py — Jarvis2 terminal UI for Linux GNOME (curses edition).

Layout (curses):
  ┌─────────────────────────────────────────────┐
  │  ╔═══════════════════════════════════════╗  │  ← banner (3 lines, drawn once)
  │  ║          J.A.R.V.I.S  —  Linux        ║  │
  │  ╚═══════════════════════════════════════╝  │
  ├── [STATUS] ──────────────────────────────── │  ← status bar (1 line, always row 4)
  │                                             │
  │   (scrollable conversation log)             │  ← log area (dynamic height)
  │                                             │
  ├─────────────────────────────────────────────│  ← separator
  │  ❯  _                                       │  ← STATIC input box (last 3 rows)
  └─────────────────────────────────────────────┘

The input box never moves — it is anchored to the bottom of the terminal.
Typing is handled by curses (character-by-character), so no readline flicker.

Public API (same as original JarvisUI):
    set_state(state)
    write_log(message)
    toggle_mute()
    start_input_loop()          ← launches the curses wrapper in a thread
    shutdown()
    muted : bool
    on_text_command : Callable[[str], None]
"""
from __future__ import annotations

import curses
import threading
import time
from datetime import datetime
from typing import Callable


# ── State / label tables ───────────────────────────────────────────────────────
_STATES: dict[str, tuple[str, int]] = {
    # state        label              curses color-pair index
    "LISTENING": ("◉  LISTENING",   1),   # green
    "SPEAKING":  ("▶  SPEAKING",    2),   # cyan
    "THINKING":  ("◌  THINKING",    3),   # yellow
    "MUTED":     ("✕  MUTED",       4),   # red
    "OFFLINE":   ("○  OFFLINE",     5),   # dim white
}

_LOG_TAGS: dict[str, tuple[str, int]] = {
    "You":    ("You   ", 1),   # green
    "Jarvis": ("Jarvis", 2),   # cyan
    "SYS":    ("System", 5),   # dim
    "ERR":    ("Error ", 4),   # red
}

# Pair indices → (fg, bg) — assigned in _init_colors()
# 1=green 2=cyan 3=yellow 4=red 5=gray 6=blue(banner) 7=white(body)

_BANNER_LINES = [
    "  ╔══════════════════════════════════════════╗",
    "  ║               J.A.R.V.I.S                ║",
    "  ║   Just A Rather Very Intelligent System  ║",
    "  ║                   Linux                  ║",
    "  ╚══════════════════════════════════════════╝",
]
_BANNER_H = len(_BANNER_LINES)   # 5
_STATUS_ROW = _BANNER_H          # row index right after the banner
_LOG_START  = _STATUS_ROW + 1    # first log row

_INPUT_ROWS = 3    # separator + prompt + blank padding


class JarvisUI:
    """
    Curses-based terminal UI for Jarvis2.

    The window is divided into:
      • Fixed banner at the top (ASCII art, never scrolls)
      • Fixed status bar below the banner
      • Scrollable log in the middle
      • Fixed input box at the bottom (3 rows)
    """

    def __init__(self) -> None:
        self._state          = "OFFLINE"
        self._last_state     = ""
        self.muted           = False
        self.on_text_command: Callable[[str], None] | None = None

        # Log storage: list of (tag_key, message) pairs
        self._log: list[tuple[str, str]] = []
        self._log_lock = threading.Lock()

        # Status pulse
        self._pulse_stop  = threading.Event()
        self._pulse_label = ""
        self._pulse_pair  = 5
        self._pulse_lock  = threading.Lock()

        # Curses screen (set by _run_curses)
        self._scr: curses.window | None = None
        self._scr_lock = threading.Lock()
        self._rows = 24
        self._cols = 80

        # Input buffer
        self._input_buf = ""
        self._cursor_pos = 0        # character offset inside _input_buf

        # Scroll offset (0 = newest at bottom)
        self._scroll_offset = 0

        # Ready event — wait until curses is initialised before other threads draw
        self._ready = threading.Event()

        # Shutdown flag
        self._alive = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def start_input_loop(self) -> None:
        """Launch the curses main loop in a daemon thread."""
        t = threading.Thread(target=self._run_curses, daemon=True)
        t.start()
        # Let curses initialise before callers start drawing
        self._ready.wait(timeout=5)

    def set_state(self, state: str) -> None:
        with self._pulse_lock:
            if self.muted and state == "LISTENING":
                state = "MUTED"
            if state == self._last_state:
                return
            self._last_state = state
            self._state      = state
            label, pair      = _STATES.get(state, ("◌  UNKNOWN", 5))
            self._pulse_label = label
            self._pulse_pair  = pair

        self._redraw_status(label, pair)
        self._restart_pulse(label, pair)

    def write_log(self, message: str) -> None:
        """Append a formatted line to the conversation log."""
        # Detect tag
        tag = "SYS"
        body = message
        for t in _LOG_TAGS:
            if message.startswith(f"{t}:"):
                tag  = t
                body = message[len(t) + 1:].strip()
                break

        ts = datetime.now().strftime("%H:%M:%S")
        with self._log_lock:
            self._log.append((tag, ts, body))
            # Reset scroll so newest is visible
            self._scroll_offset = 0

        self._redraw_log()

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
        self._alive = False
        self._last_state = ""
        self.set_state("OFFLINE")

    # ── Curses initialisation ──────────────────────────────────────────────────

    def _run_curses(self) -> None:
        curses.wrapper(self._curses_main)

    def _curses_main(self, scr: curses.window) -> None:
        self._scr = scr
        curses.curs_set(0)          # hide system cursor — we draw our own
        curses.noecho()
        curses.raw()
        scr.keypad(True)
        scr.timeout(100)            # non-blocking getch (100 ms)

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

            if ch == curses.ERR:
                continue

            self._handle_key(ch)

    def _init_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        # pair(n, fg, bg)  — bg=-1 means terminal default
        curses.init_pair(1, curses.COLOR_GREEN,   -1)   # You / LISTENING
        curses.init_pair(2, curses.COLOR_CYAN,    -1)   # Jarvis / SPEAKING
        curses.init_pair(3, curses.COLOR_YELLOW,  -1)   # THINKING
        curses.init_pair(4, curses.COLOR_RED,     -1)   # ERR / MUTED
        curses.init_pair(5, 8,                    -1)   # gray (SYS / OFFLINE)
        curses.init_pair(6, curses.COLOR_CYAN,    -1)   # banner accent
        curses.init_pair(7, curses.COLOR_WHITE,   -1)   # body text

    # ── Full redraw ────────────────────────────────────────────────────────────

    def _full_redraw(self) -> None:
        if not self._scr:
            return
        with self._scr_lock:
            self._scr.erase()
            self._draw_banner()
            label, pair = _STATES.get(self._state, ("○  OFFLINE", 5))
            self._draw_status_line(label, pair)
            self._draw_log_region()
            self._draw_input_box()
            self._scr.refresh()

    # ── Banner ─────────────────────────────────────────────────────────────────

    def _draw_banner(self) -> None:
        scr = self._scr
        pair = curses.color_pair(6) | curses.A_BOLD
        for i, line in enumerate(_BANNER_LINES):
            self._safe_addstr(i, 0, line[:self._cols], pair)

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _redraw_status(self, label: str, pair_idx: int) -> None:
        if not self._scr:
            return
        with self._scr_lock:
            self._draw_status_line(label, pair_idx)
            self._scr.refresh()

    def _draw_status_line(self, label: str, pair_idx: int) -> None:
        scr   = self._scr
        row   = _STATUS_ROW
        ts    = datetime.now().strftime("%H:%M:%S")
        line  = f"  {ts}  {label}"
        # Fill rest of row with spaces to clear previous content
        line  = line.ljust(self._cols - 1)
        self._safe_addstr(row, 0, line[:self._cols - 1],
                          curses.color_pair(pair_idx) | curses.A_BOLD)

    # ── Log region ─────────────────────────────────────────────────────────────

    def _log_area_height(self) -> int:
        """Available rows between status bar and input box."""
        return max(1, self._rows - _LOG_START - _INPUT_ROWS)

    def _redraw_log(self) -> None:
        if not self._scr:
            return
        with self._scr_lock:
            self._draw_log_region()
            self._scr.refresh()

    def _draw_log_region(self) -> None:
        scr    = self._scr
        height = self._log_area_height()
        width  = self._cols

        with self._log_lock:
            entries = list(self._log)
            offset  = self._scroll_offset

        # Render each entry into display lines (with word-wrap)
        display_lines: list[tuple[int, str, str]] = []   # (pair, prefix, body)
        for tag, ts, body in entries:
            pfx_text, pair_idx = _LOG_TAGS.get(tag, ("      ", 7))
            separator = "│"
            prefix_str = f"  {ts}  {pfx_text}  {separator}  "
            # Simple word-wrap for body
            max_body = max(1, width - len(prefix_str) - 1)
            words     = body.split()
            line_acc  = ""
            first     = True
            if not words:
                display_lines.append((pair_idx, prefix_str if first else " " * len(prefix_str), ""))
                continue
            for word in words:
                if len(line_acc) + (1 if line_acc else 0) + len(word) <= max_body:
                    line_acc += (" " if line_acc else "") + word
                else:
                    display_lines.append((pair_idx,
                                          prefix_str if first else " " * len(prefix_str),
                                          line_acc))
                    first    = False
                    line_acc = word
            if line_acc:
                display_lines.append((pair_idx,
                                      prefix_str if first else " " * len(prefix_str),
                                      line_acc))

        # Slice for visible area (newest = bottom)
        total = len(display_lines)
        start = max(0, total - height - offset)
        end   = max(0, total - offset)
        visible = display_lines[start:end]

        # Pad top if not enough lines
        blank_rows = height - len(visible)

        for r in range(height):
            row = _LOG_START + r
            if r < blank_rows:
                self._safe_addstr(row, 0, " " * (width - 1), curses.color_pair(7))
            else:
                idx = r - blank_rows
                pair_idx, prefix_str, body = visible[idx]
                # Draw prefix in dim, body in color
                full_line = (prefix_str + body).ljust(width - 1)
                self._safe_addstr(row, 0, full_line[:width - 1],
                                  curses.color_pair(7) | curses.A_DIM)
                # Overdraw colorful parts
                if len(prefix_str) < width:
                    pfx_display = prefix_str[:width - 1]
                    self._safe_addstr(row, 0, pfx_display,
                                      curses.color_pair(pair_idx) | curses.A_BOLD)

    # ── Input box (static, pinned to bottom) ──────────────────────────────────

    def _draw_input_box(self) -> None:
        scr   = self._scr
        width = self._cols
        rows  = self._rows

        sep_row    = rows - _INPUT_ROWS        # separator line
        prompt_row = rows - _INPUT_ROWS + 1    # actual input line
        hint_row   = rows - 1                  # hint / controls

        # ── Separator ──────────────────────────────────────────────────────────
        sep = "─" * (width - 1)
        self._safe_addstr(sep_row, 0, sep, curses.color_pair(5))

        # ── Prompt line ────────────────────────────────────────────────────────
        prompt   = "  ❯  "
        prompt_p = curses.color_pair(2) | curses.A_BOLD
        self._safe_addstr(prompt_row, 0, prompt, prompt_p)

        # Draw typed buffer
        buf_start = len(prompt)
        buf_area  = width - buf_start - 1
        buf_text  = self._input_buf
        # If buffer wider than area, show tail
        if len(buf_text) > buf_area:
            buf_text = buf_text[-(buf_area):]
        padded = buf_text.ljust(buf_area)
        self._safe_addstr(prompt_row, buf_start, padded[:buf_area], curses.color_pair(7))

        # Draw cursor block
        cursor_col = buf_start + min(self._cursor_pos, buf_area - 1)
        char_under = padded[min(self._cursor_pos, len(padded) - 1)] if padded else " "
        self._safe_addstr(prompt_row, cursor_col, char_under,
                          curses.color_pair(3) | curses.A_REVERSE)

        # ── Hint line ──────────────────────────────────────────────────────────
        hint = "  [Enter] send  [m] mute  [↑/↓] scroll  [Ctrl+C] exit"
        self._safe_addstr(hint_row, 0, hint[:width - 1], curses.color_pair(5))

    def _refresh_input_box(self) -> None:
        if not self._scr:
            return
        with self._scr_lock:
            self._draw_input_box()
            self._scr.refresh()

    # ── Key handling ───────────────────────────────────────────────────────────

    def _handle_key(self, ch: int) -> None:
        if ch in (curses.KEY_ENTER, 10, 13):          # Enter
            self._submit_input()
        elif ch in (curses.KEY_BACKSPACE, 127, 8):    # Backspace
            if self._cursor_pos > 0:
                pos = self._cursor_pos
                self._input_buf = self._input_buf[:pos - 1] + self._input_buf[pos:]
                self._cursor_pos -= 1
            self._refresh_input_box()
        elif ch == curses.KEY_LEFT:
            if self._cursor_pos > 0:
                self._cursor_pos -= 1
            self._refresh_input_box()
        elif ch == curses.KEY_RIGHT:
            if self._cursor_pos < len(self._input_buf):
                self._cursor_pos += 1
            self._refresh_input_box()
        elif ch == curses.KEY_UP:
            self._scroll_offset += 1
            self._redraw_log()
            self._refresh_input_box()
        elif ch == curses.KEY_DOWN:
            self._scroll_offset = max(0, self._scroll_offset - 1)
            self._redraw_log()
            self._refresh_input_box()
        elif ch == curses.KEY_HOME:
            self._cursor_pos = 0
            self._refresh_input_box()
        elif ch == curses.KEY_END:
            self._cursor_pos = len(self._input_buf)
            self._refresh_input_box()
        elif ch == curses.KEY_DC:                     # Delete
            pos = self._cursor_pos
            if pos < len(self._input_buf):
                self._input_buf = self._input_buf[:pos] + self._input_buf[pos + 1:]
            self._refresh_input_box()
        elif ch == 3:                                 # Ctrl+C
            self.shutdown()
            raise KeyboardInterrupt
        elif 32 <= ch < 256:                          # Printable ASCII
            pos = self._cursor_pos
            self._input_buf = self._input_buf[:pos] + chr(ch) + self._input_buf[pos:]
            self._cursor_pos += 1
            self._refresh_input_box()

    def _submit_input(self) -> None:
        cmd = self._input_buf.strip()
        self._input_buf  = ""
        self._cursor_pos = 0
        self._refresh_input_box()

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
            self._redraw_status(label + suffix, pair_idx)
            i += 1

    # ── Curses safe helper ─────────────────────────────────────────────────────

    def _safe_addstr(self, row: int, col: int, text: str, attr: int = 0) -> None:
        """Draw text without raising on out-of-bounds cells."""
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