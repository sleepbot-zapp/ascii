from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from asciify.core.pipeline import AsciiFrame

DEFAULT_FONT_SIZE = 12
DEFAULT_FRAME_DURATION_MS = 100
DEFAULT_CELL_BG_FACTOR = 0.35


def _muted_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(round(v * factor)) for v in color)


def _make_pil_image(
    ascii_frame: AsciiFrame,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    char_w: int,
    char_h: int,
    background: tuple[int, int, int],
    foreground: tuple[int, int, int],
    cell_bg_factor: float = DEFAULT_CELL_BG_FACTOR,
) -> Image.Image:
    rows, cols = ascii_frame.chars.shape
    img = Image.new("RGB", (cols * char_w, rows * char_h), background)
    draw = ImageDraw.Draw(img)
    for r in range(rows):
        y0 = r * char_h
        for c in range(cols):
            x0 = c * char_w
            if ascii_frame.colors is not None:
                color = tuple(int(v) for v in ascii_frame.colors[r, c])
                draw.rectangle(
                    (x0, y0, x0 + char_w, y0 + char_h),
                    fill=_muted_color(color, cell_bg_factor),
                )
            else:
                color = foreground
            draw.text(
                (x0, y0),
                ascii_frame.chars[r, c],
                fill=color,
                font=font,
            )
    return img


def _load_font(
    font_path: str | None, font_size: int
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]:
    font = (
        ImageFont.truetype(font_path, font_size)
        if font_path
        else ImageFont.load_default(size=font_size)
    )
    bbox = font.getbbox("M")
    char_w, char_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return font, char_w, char_h


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


def render_image(
    ascii_frame: AsciiFrame,
    output_path: str | Path,
    *,
    font_path: str | None = None,
    font_size: int = DEFAULT_FONT_SIZE,
    background: tuple[int, int, int] = (255, 255, 255),
    foreground: tuple[int, int, int] = (0, 0, 0),
) -> None:
    font, char_w, char_h = _load_font(font_path, font_size)
    img = _make_pil_image(ascii_frame, font, char_w, char_h, background, foreground)
    img.save(output_path)


def render_gif(
    ascii_frames: Iterable[AsciiFrame],
    output_path: str | Path,
    *,
    font_path: str | None = None,
    font_size: int = DEFAULT_FONT_SIZE,
    background: tuple[int, int, int] = (255, 255, 255),
    foreground: tuple[int, int, int] = (0, 0, 0),
    loop: int = 0,
) -> None:
    font, char_w, char_h = _load_font(font_path, font_size)

    pil_frames: list[Image.Image] = []
    durations: list[int] = []

    for af in ascii_frames:
        pil_frames.append(
            _make_pil_image(af, font, char_w, char_h, background, foreground)
        )
        durations.append(
            int(af.duration_ms)
            if af.duration_ms is not None
            else DEFAULT_FRAME_DURATION_MS
        )

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


def render_video(
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
) -> None:
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "render_video() requires PyAV.  Install it with:  pip install av"
        ) from exc

    from fractions import Fraction

    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    if suffix == ".webm":
        codec_name = "vp9"
        pix_fmt = "yuv420p"
    else:
        codec_name = "libx264"
        pix_fmt = "yuv420p"

    font, char_w, char_h = _load_font(font_path, font_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps_fraction = Fraction(fps).limit_denominator(1001)

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

        for af in ascii_frames:
            pil_img = _make_pil_image(af, font, char_w, char_h, background, foreground)

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

        if not width_set:
            raise ValueError("ascii_frames yielded no frames; nothing to write.")

        for packet in stream.encode():
            container.mux(packet)

    finally:
        container.close()
