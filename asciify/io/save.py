from __future__ import annotations

import queue
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from asciify.core.pipeline import AsciiFrame


DEFAULT_FONT_SIZE = 12
DEFAULT_FRAME_DURATION_MS = 100
_SENTINEL = object()


def _load_font(
    font_path: str | None,
    font_size: int,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]:
    font = (
        ImageFont.truetype(font_path, font_size)
        if font_path
        else ImageFont.load_default(size=font_size)
    )
    bbox = font.getbbox("M")
    char_w, char_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return font, char_w, char_h


def _heuristic_bg_fg(
    pixels: np.ndarray,
    *,
    invert: bool = False,
    sample_frac: float = 0.05,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    flat = pixels.reshape(-1, 3)
    rng = np.random.default_rng(seed=0)
    n_sample = max(1, int(len(flat) * sample_frac))
    idx = rng.choice(len(flat), size=n_sample, replace=False)
    sample = flat[idx].astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum = sample @ weights
    n10 = max(1, n_sample // 10)
    order = np.argsort(lum)
    dark_rgb = np.median(sample[order[:n10]], axis=0).round().astype(int)
    light_rgb = np.median(sample[order[-n10:]], axis=0).round().astype(int)

    def _clamp(arr):
        return (
            int(np.clip(arr[0], 0, 255)),
            int(np.clip(arr[1], 0, 255)),
            int(np.clip(arr[2], 0, 255)),
        )

    dark, light = _clamp(dark_rgb), _clamp(light_rgb)
    return (dark, light) if invert else (light, dark)


_cnn_model = None
_cnn_loaded = False
_CNN_INPUT_SIZE = 64


def _load_cnn(model_path=None):
    global _cnn_model, _cnn_loaded
    if _cnn_loaded:
        return _cnn_model
    _cnn_loaded = True

    import os

    if model_path is None:
        env = os.environ.get("ASCIIFY_BGFG_MODEL")
        if env:
            candidate = Path(env)
        else:
            candidate = Path(__file__).with_name("bgfg_cnn.pt")
    else:
        candidate = Path(model_path)

    if not candidate.is_file():
        return None

    try:
        import torch

        _cnn_model = torch.jit.load(str(candidate), map_location="cpu")
        _cnn_model.eval()
        return _cnn_model
    except Exception:
        return None


def sample_bg_fg(
    frame,
    *,
    invert: bool = False,
    model_path=None,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    cnn = _load_cnn(model_path)

    if cnn is not None:
        try:
            import torch
            import torchvision.transforms.functional as TF

            pil = Image.fromarray(frame.pixels, mode="RGB").resize(
                (_CNN_INPUT_SIZE, _CNN_INPUT_SIZE), Image.Resampling.BILINEAR
            )
            x = TF.to_tensor(pil).unsqueeze(0)

            with torch.no_grad():
                pred = cnn(x)[0].numpy()

            bg = tuple(int(round(v * 255)) for v in pred[:3])
            fg = tuple(int(round(v * 255)) for v in pred[3:])

            if invert:
                bg, fg = fg, bg

            return bg, fg

        except Exception:
            pass

    return _heuristic_bg_fg(frame.pixels, invert=invert)


class FastFrameRenderer:
    def __init__(
        self,
        font_path: str | None = None,
        font_size: int = DEFAULT_FONT_SIZE,
        background: tuple[int, int, int] = (0, 0, 0),
        foreground: tuple[int, int, int] = (255, 255, 255),
        cell_bg_factor: float = 0.35,
    ) -> None:
        self.font, self.char_w, self.char_h = _load_font(font_path, font_size)
        self.background = background
        self.foreground = foreground

        self.cell_bg_factor = cell_bg_factor

        self._stamp_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _muted_color(
        color: tuple[int, int, int], factor: float
    ) -> tuple[int, int, int]:
        return tuple(int(round(v * factor)) for v in color)

    def _make_stamp(
        self,
        char: str,
        color: tuple[int, int, int],
        *,
        cell_bg: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        backdrop = cell_bg if cell_bg is not None else self.background
        img = Image.new("RGB", (self.char_w, self.char_h), backdrop)
        ImageDraw.Draw(img).text((0, 0), char, fill=color, font=self.font)
        return np.asarray(img, dtype=np.uint8)

    def _get_mono_stamp(self, char: str) -> np.ndarray:
        if char not in self._stamp_cache:
            self._stamp_cache[char] = self._make_stamp(char, self.foreground)
        return self._stamp_cache[char]

    def render(self, ascii_frame: AsciiFrame) -> Image.Image:
        rows, cols = ascii_frame.chars.shape
        canvas = np.full(
            (rows * self.char_h, cols * self.char_w, 3),
            self.background,
            dtype=np.uint8,
        )

        if ascii_frame.colors is not None:
            for r in range(rows):
                y0 = r * self.char_h
                y1 = y0 + self.char_h
                for c in range(cols):
                    ch = ascii_frame.chars[r, c]
                    color = tuple(int(v) for v in ascii_frame.colors[r, c])

                    key = (ch, color)
                    if key not in self._stamp_cache:
                        cell_bg = self._muted_color(color, self.cell_bg_factor)
                        self._stamp_cache[key] = self._make_stamp(
                            ch, color, cell_bg=cell_bg
                        )
                    stamp = self._stamp_cache[key]
                    x0 = c * self.char_w
                    canvas[y0:y1, x0 : x0 + self.char_w] = stamp
        else:
            unique_chars = np.unique(ascii_frame.chars)
            for ch in unique_chars:
                stamp = self._get_mono_stamp(str(ch))
                positions = np.argwhere(ascii_frame.chars == ch)
                for r, c in positions:
                    y0 = r * self.char_h
                    x0 = c * self.char_w
                    canvas[y0 : y0 + self.char_h, x0 : x0 + self.char_w] = stamp

        return Image.fromarray(canvas, mode="RGB")


ProgressCallback = Callable[[int, int | None, AsciiFrame], None]


def _build_shared_palette(
    pil_frames: list[Image.Image],
    *,
    max_colors: int = 256,
    max_sample_frames: int = 12,
) -> Image.Image:
    if not pil_frames:
        raise ValueError("pil_frames is empty; nothing to build a palette from.")

    step = max(1, len(pil_frames) // max_sample_frames)
    sample = pil_frames[::step] or pil_frames[:1]

    w, h = sample[0].size
    composite = Image.new("RGB", (w, h * len(sample)))
    for i, frame in enumerate(sample):
        composite.paste(frame, (0, i * h))

    return composite.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)


def _iter_with_count(
    source: Iterable[AsciiFrame],
    out_q: "queue.Queue[object]",
    error_box: list[BaseException],
) -> None:
    try:
        for af in source:
            out_q.put(af)
    except Exception as exc:
        error_box.append(exc)
    finally:
        out_q.put(_SENTINEL)


def save_video(
    ascii_frames: Iterable[AsciiFrame],
    output_path: str | Path,
    *,
    fps: float = 25.0,
    font_path: str | None = None,
    font_size: int = DEFAULT_FONT_SIZE,
    background: tuple[int, int, int] = (0, 0, 0),
    foreground: tuple[int, int, int] = (255, 255, 255),
    crf: int = 18,
    preset: str = "medium",
    total_frames: int | None = None,
    on_progress: ProgressCallback | None = None,
    queue_depth: int = 8,
) -> None:
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "save_video() requires PyAV.  Install it with:  pip install av"
        ) from exc

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    codec_name = "vp9" if suffix == ".webm" else "libx264"
    pix_fmt = "yuv420p"

    renderer = FastFrameRenderer(
        font_path=font_path,
        font_size=font_size,
        background=background,
        foreground=foreground,
    )

    frame_q: queue.Queue[object] = queue.Queue(maxsize=queue_depth)
    error_box: list[BaseException] = []
    producer = threading.Thread(
        target=_iter_with_count,
        args=(ascii_frames, frame_q, error_box),
        daemon=True,
    )
    producer.start()

    fps_fraction = Fraction(fps).limit_denominator(1001)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(output_path), mode="w")
    try:
        stream = container.add_stream(codec_name, rate=fps_fraction)
        stream.pix_fmt = pix_fmt
        if codec_name == "libx264":
            stream.options = {"crf": str(crf), "preset": preset}
        elif codec_name == "vp9":
            stream.options = {"crf": str(crf), "b:v": "0"}

        frame_index = 0
        width_set = False

        while True:
            item = frame_q.get()
            if item is _SENTINEL:
                break

            af: AsciiFrame = item

            pil_img = renderer.render(af)

            if not width_set:
                w, h = pil_img.size
                stream.width = w + (w % 2)
                stream.height = h + (h % 2)
                width_set = True

            w, h = pil_img.size
            if w != stream.width or h != stream.height:
                padded = Image.new("RGB", (stream.width, stream.height), background)
                padded.paste(pil_img)
                pil_img = padded

            av_frame = av.VideoFrame.from_image(pil_img)
            av_frame = av_frame.reformat(format=pix_fmt)
            av_frame.pts = frame_index

            for packet in stream.encode(av_frame):
                container.mux(packet)

            frame_index += 1

            if on_progress is not None:
                on_progress(frame_index, total_frames, af)

        if not width_set:
            raise ValueError("ascii_frames yielded no frames; nothing to write.")

        for packet in stream.encode():
            container.mux(packet)

    finally:
        container.close()

    producer.join()
    if error_box:
        raise error_box[0]


def save_gif(
    ascii_frames: Iterable[AsciiFrame],
    output_path: str | Path,
    *,
    font_path: str | None = None,
    font_size: int = DEFAULT_FONT_SIZE,
    background: tuple[int, int, int] = (255, 255, 255),
    foreground: tuple[int, int, int] = (0, 0, 0),
    loop: int = 0,
    total_frames: int | None = None,
    on_progress: ProgressCallback | None = None,
    queue_depth: int = 8,
) -> None:
    renderer = FastFrameRenderer(
        font_path=font_path,
        font_size=font_size,
        background=background,
        foreground=foreground,
    )

    frame_q: queue.Queue[object] = queue.Queue(maxsize=queue_depth)
    error_box: list[BaseException] = []
    producer = threading.Thread(
        target=_iter_with_count,
        args=(ascii_frames, frame_q, error_box),
        daemon=True,
    )
    producer.start()

    pil_frames: list[Image.Image] = []
    durations: list[int] = []
    done = 0

    while True:
        item = frame_q.get()
        if item is _SENTINEL:
            break

        af: AsciiFrame = item
        pil_frames.append(renderer.render(af))
        durations.append(
            int(af.duration_ms)
            if af.duration_ms is not None
            else DEFAULT_FRAME_DURATION_MS
        )
        done += 1

        if on_progress is not None:
            on_progress(done, total_frames, af)

    producer.join()
    if error_box:
        raise error_box[0]

    if not pil_frames:
        raise ValueError("ascii_frames yielded no frames; nothing to write.")

    palette_img = _build_shared_palette(pil_frames)
    palette_frames = [
        f.quantize(palette=palette_img, dither=Image.Dither.NONE) for f in pil_frames
    ]
    palette_frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=loop,
        optimize=False,
    )
