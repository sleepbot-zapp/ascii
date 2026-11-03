from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from asciify.core.frame import Frame
from PIL import Image


DEFAULT_CHAR_ASPECT = 2.0


@dataclass(frozen=True, slots=True)
class CellGrid:
    rgb: np.ndarray

    @property
    def rows(self) -> int:
        return self.rgb.shape[0]

    @property
    def cols(self) -> int:
        return self.rgb.shape[1]

    def luminance(self) -> np.ndarray:
        weights = np.array([0.299, 0.587, 0.114])
        return (self.rgb.astype(np.float64) @ weights) / 255.0


def compute_rows(
    width: int, height: int, cols: int, char_aspect: float = DEFAULT_CHAR_ASPECT
) -> int:
    return max(1, round((height / width) * cols / char_aspect))


def to_cell_grid(
    frame: Frame, cols: int, char_aspect: float = DEFAULT_CHAR_ASPECT
) -> CellGrid:
    rows = compute_rows(frame.width, frame.height, cols, char_aspect)
    img = Image.fromarray(frame.pixels, mode="RGB")
    resized = img.resize((cols, rows), resample=Image.Resampling.BOX)
    return CellGrid(rgb=np.asarray(resized, dtype=np.uint8))
