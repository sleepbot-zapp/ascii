from __future__ import annotations

from asciify.core.pipeline import AsciiFrame

RESET = "\x1b[0m"


def render_terminal(ascii_frame: AsciiFrame) -> str:
    rows, cols = ascii_frame.chars.shape
    lines = []
    for r in range(rows):
        if ascii_frame.colors is None:
            lines.append("".join(ascii_frame.chars[r]))
            continue
        parts = []
        for c in range(cols):
            red, green, blue = ascii_frame.colors[r, c]
            parts.append(f"\x1b[38;2;{red};{green};{blue}m{ascii_frame.chars[r, c]}")
        parts.append(RESET)
        lines.append("".join(parts))
    return "\n".join(lines)
