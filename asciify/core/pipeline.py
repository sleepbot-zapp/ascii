from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asciify.core.frame import Frame
from asciify.core.glyphs import DensityRamp, GlyphSelector
from asciify.core.sampling import DEFAULT_CHAR_ASPECT, CellGrid, to_cell_grid


@dataclass(frozen=True, slots=True)
class AsciiFrame:
    chars: np.ndarray
    colors: np.ndarray | None
    source_index: int = 0
    duration_ms: float | None = None


def _density_ramp_fast(
    ramp_chars: str, luminance: np.ndarray, invert: bool
) -> np.ndarray:
    n = len(ramp_chars)

    t = luminance if invert else (1.0 - luminance)

    indices = np.clip(np.round(t * (n - 1)).astype(int), 0, n - 1)

    char_array = np.array(list(ramp_chars), dtype="U1")
    return char_array[indices]


class Pipeline:
    def __init__(
        self,
        *,
        cols: int = 100,
        char_aspect: float = DEFAULT_CHAR_ASPECT,
        glyph_selector: GlyphSelector | None = None,
        color: bool = False,
    ) -> None:
        self.cols = cols
        self.char_aspect = char_aspect
        self.glyph_selector: GlyphSelector = glyph_selector or DensityRamp()
        self.color = color

    def process(self, frame: Frame) -> AsciiFrame:
        grid: CellGrid = to_cell_grid(frame, self.cols, self.char_aspect)
        luminance = grid.luminance()

        if isinstance(self.glyph_selector, DensityRamp):
            chars = _density_ramp_fast(
                self.glyph_selector.ramp,
                luminance,
                self.glyph_selector.invert,
            )
        else:
            chars = np.vectorize(self.glyph_selector.select)(luminance)

        colors = grid.rgb if self.color else None
        return AsciiFrame(
            chars=chars,
            colors=colors,
            source_index=frame.index,
            duration_ms=frame.duration_ms,
        )
