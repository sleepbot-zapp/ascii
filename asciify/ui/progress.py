from __future__ import annotations

import sys
import time
from pathlib import Path

from core import AsciiFrame
from render import render_text, render_terminal



_ESC = "\x1b["
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
GREEN = "\x1b[32m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
WHITE = "\x1b[97m"

FILLED = "█"
PARTIAL = "░"
EMPTY = " "


BAR_WIDTH = 36


def _move_up(n: int) -> str:
    return f"{_ESC}{n}A" if n > 0 else ""


def _clear_line() -> str:
    return f"{_ESC}2K\r"


def _fmt_seconds(s: float) -> str:
    s = int(s)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _progress_bar(fraction: float, width: int = BAR_WIDTH) -> str:
    filled = round(fraction * width)
    bar = FILLED * filled + PARTIAL * (width - filled)
    pct = int(fraction * 100)
    return f"[{GREEN}{bar}{RESET}] {BOLD}{WHITE}{pct:3d}%{RESET}"


class ProgressUI:
    HEADER_LINES = 3

    def __init__(
        self,
        output_path: str | Path,
        *,
        total_frames: int | None = None,
        color: bool = False,
        preview_rows: int | None = 20,
    ) -> None:
        self.output_path = Path(output_path)
        self.total_frames = total_frames
        self.color = color
        self.preview_rows = preview_rows

        self._done = 0
        self._start = time.monotonic()
        self._fps_window: list[float] = []
        self._fps_window_size = 30
        self._last_preview_lines = 0
        self._initialized = False

        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()

    def on_frame(
        self,
        frames_done: int,
        total_frames: int | None,
        latest_frame: AsciiFrame,
    ) -> None:
        if total_frames is not None:
            self.total_frames = total_frames

        self._done = frames_done
        now = time.monotonic()
        self._fps_window.append(now)
        if len(self._fps_window) > self._fps_window_size:
            self._fps_window.pop(0)

        self._redraw(latest_frame, now)

    def finish(self) -> None:
        elapsed = time.monotonic() - self._start
        self._erase_preview()
        self._erase_header()

        n = self._done
        t = _fmt_seconds(elapsed)
        avg_fps = n / elapsed if elapsed > 0 else 0

        sys.stdout.write(
            f"\n{GREEN}{BOLD}✓ Saved{RESET} {self.output_path.name}  "
            f"— {n} frames in {t}  ({avg_fps:.1f} fps avg)\n\n"
        )
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

    def _current_fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        return (len(self._fps_window) - 1) / span if span > 0 else 0.0

    def _eta(self, elapsed: float) -> str | None:
        if self.total_frames is None or self._done == 0:
            return None
        fps = self._current_fps()
        if fps <= 0:
            return None
        remaining = (self.total_frames - self._done) / fps
        return _fmt_seconds(remaining)

    def _header_lines(self, elapsed: float) -> list[str]:
        name = self.output_path.name
        fps = self._current_fps()
        elapsed_str = _fmt_seconds(elapsed)

        line1 = f"{BOLD}{CYAN}Saving{RESET} {WHITE}{name}{RESET}"

        if self.total_frames:
            frac = min(1.0, self._done / self.total_frames)
            bar = _progress_bar(frac)
            eta = self._eta(elapsed)
            eta_str = f"  ETA {YELLOW}{eta}{RESET}" if eta else ""
            line2 = f"  {bar}{eta_str}"
        else:
            spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            ch = spinner[self._done % len(spinner)]
            line2 = f"  {CYAN}{ch}{RESET}  processing…"

        total_str = f"/ {self.total_frames}" if self.total_frames else ""
        fps_str = f"{fps:.1f} fps" if fps > 0 else "-- fps"
        line3 = (
            f"  {DIM}Frame {self._done} {total_str}   •  "
            f"{fps_str}   •  elapsed {elapsed_str}{RESET}"
        )

        return [line1, line2, line3]

    def _erase_header(self) -> None:
        for _ in range(self.HEADER_LINES):
            sys.stdout.write(_clear_line())
            sys.stdout.write(f"{_ESC}1A")

        sys.stdout.write(f"{_ESC}{self.HEADER_LINES}B")

    def _erase_preview(self) -> None:
        if self._last_preview_lines == 0:
            return

        sys.stdout.write(_move_up(self._last_preview_lines))
        for _ in range(self._last_preview_lines):
            sys.stdout.write(_clear_line())
            sys.stdout.write(f"{_ESC}1B")
        sys.stdout.write(_move_up(self._last_preview_lines))

    def _redraw(self, frame: AsciiFrame, now: float) -> None:
        elapsed = now - self._start

        if not self._initialized:
            for _ in range(self.HEADER_LINES):
                sys.stdout.write("\n")
            self._initialized = True

        total_up = self.HEADER_LINES + self._last_preview_lines
        sys.stdout.write(_move_up(total_up))

        for line in self._header_lines(elapsed):
            sys.stdout.write(_clear_line() + line + "\n")

        preview_text = (
            render_terminal(frame) if self.color else render_text(frame)
        )
        preview_lines = preview_text.split("\n")

        if self.preview_rows is not None:
            preview_lines = preview_lines[: self.preview_rows]

        for line in preview_lines:
            sys.stdout.write(_clear_line() + line + "\n")

        self._last_preview_lines = len(preview_lines)
        sys.stdout.flush()
