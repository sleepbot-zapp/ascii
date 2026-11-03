from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True, slots=True)
class Frame:
    pixels: np.ndarray
    index: int = 0
    duration_ms: float | None = None

    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise ValueError(
                f"Frame.pixels must have shape (H, W, 3), got {self.pixels.shape}"
            )
        if self.pixels.dtype != np.uint8:
            raise ValueError(f"Frame.pixels must be uint8, got {self.pixels.dtype}")

    @property
    def height(self) -> int:
        return self.pixels.shape[0]

    @property
    def width(self) -> int:
        return self.pixels.shape[1]
