from __future__ import annotations

from asciify.core.pipeline import AsciiFrame


def render_text(ascii_frame: AsciiFrame) -> str:
    return "\n".join("".join(row) for row in ascii_frame.chars)
