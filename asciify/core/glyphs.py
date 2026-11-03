from __future__ import annotations

from typing import Protocol, runtime_checkable


RAMPS: dict[str, str] = {
    "block": " .:-=+*#%@",
    "standard": (
        " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
    ),
    "minimal": " .oO@",
    "braille": " ░▒▓█",
}

DEFAULT_RAMP = "standard"


@runtime_checkable
class GlyphSelector(Protocol):
    def select(self, luminance: float) -> str: ...  # hope you dont break anything either


class DensityRamp:
    def __init__(
        self,
        ramp: str = DEFAULT_RAMP,
        *,
        invert: bool = False,
    ) -> None:
        self.ramp: str = RAMPS.get(ramp, ramp)
        self.invert = invert
        if not self.ramp:
            raise ValueError("ramp must be a non-empty string or a key in RAMPS")

    def select(self, luminance: float) -> str:
        n = len(self.ramp)
        t = luminance if self.invert else (1.0 - luminance)
        index = max(0, min(n - 1, round(t * (n - 1))))
        return self.ramp[index]

    def __repr__(self) -> str:
        return f"DensityRamp(ramp={self.ramp!r}, invert={self.invert!r})"
