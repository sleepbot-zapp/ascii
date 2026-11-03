from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asciify.core.glyphs import RAMPS, DensityRamp
from asciify.core.pipeline import Pipeline
from asciify.io.sources import open_source
from asciify.render import render_terminal, render_text, render_image

_ANIMATED_SUFFIXES = {
    ".gif",
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".m4v",
    ".flv",
    ".wmv",
    ".ts",
}

_VIDEO_OUT_SUFFIXES = {".mp4", ".mkv", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asciify",
        description="Turn an image, animated GIF, or video into ASCII art.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a PNG/JPEG/GIF/MP4/MKV/… file.",
    )
    parser.add_argument(
        "-c",
        "--cols",
        type=int,
        default=100,
        help="Output width in characters (default: 100).",
    )
    parser.add_argument(
        "--ramp",
        default="standard",
        choices=sorted(RAMPS),
        help="Character density ramp to use (default: standard).",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Use truecolor ANSI codes (terminal output only).",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert for light-on-dark display.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Write to this path instead of printing to stdout. "
            "Use .txt for plain text, .png for a rasterised image, "
            ".gif for an animated GIF (multi-frame inputs only)."
        ),
    )
    parser.add_argument(
        "--bg",
        default="ffffff",
        metavar="RRGGBB",
        help="Background colour as a hex triplet (default: ffffff — white).",
    )
    parser.add_argument(
        "--fg",
        default="000000",
        metavar="RRGGBB",
        help="Foreground/text colour as a hex triplet (default: 000000 — black).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Hide the live ASCII preview during saving (faster on slow terminals).",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=20,
        help="Max rows of the live preview shown while saving (default: 20).",
    )

    video_group = parser.add_argument_group("video / GIF options")
    video_group.add_argument(
        "--max-fps",
        type=float,
        default=None,
        help="Cap the frame rate for video inputs (e.g. --max-fps 10).",
    )
    video_group.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Hard limit on the number of frames to process.",
    )
    video_group.add_argument(
        "--font-size",
        type=int,
        default=12,
        help="Font size when rendering to PNG, GIF, or video (default: 12).",
    )
    video_group.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Frame rate for video output (default: 25). Ignored for GIF/PNG.",
    )
    video_group.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264/VP9 quality (0-51, lower = better; default: 18).",
    )
    video_group.add_argument(
        "--preset",
        default="medium",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="H.264 encoding speed preset (default: medium). Ignored for WebM.",
    )
    video_group.add_argument(
        "--play",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Terminal playback only (animated inputs, no --output): loop "
            "the animation N times then stop. Omit to loop forever until "
            "interrupted with Ctrl+C."
        ),
    )

    return parser


def _probe_frame_count(
    path: Path, max_fps: float | None, max_frames: int | None
) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".gif":
        try:
            from PIL import Image

            with Image.open(path) as img:
                return getattr(img, "n_frames", 1)
        except Exception:
            return None
    if suffix in _VIDEO_OUT_SUFFIXES or suffix in _ANIMATED_SUFFIXES:
        try:
            import av  # type: ignore

            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                total = stream.frames or None
                if total and max_frames:
                    total = min(total, max_frames)
                if total and max_fps and stream.average_rate:
                    native_fps = float(stream.average_rate)
                    if native_fps > max_fps:
                        total = int(total * max_fps / native_fps)
                return total
        except Exception:
            return None
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.play is not None and args.play < 1:
        build_parser().error("--play must be a positive integer.")

    selector = DensityRamp(ramp=args.ramp, invert=args.invert)
    pipeline = Pipeline(cols=args.cols, glyph_selector=selector, color=args.color)

    is_animated = args.input.suffix.lower() in _ANIMATED_SUFFIXES

    if not is_animated or (args.output and args.output.suffix.lower() == ".png"):
        source = open_source(
            args.input, max_fps=args.max_fps, max_frames=args.max_frames
        )
        frame = next(iter(source))
        ascii_frame = pipeline.process(frame)

        if args.output is None:
            text = (
                render_terminal(ascii_frame)
                if args.color
                else render_text(ascii_frame)
            )
            print(text)
        elif args.output.suffix.lower() == ".png":
            from asciify.io.save import sample_bg_fg

            background, foreground = sample_bg_fg(frame, invert=args.invert)
            render_image(
                ascii_frame,
                args.output,
                font_size=args.font_size,
                background=background,
                foreground=foreground,
            )
        else:
            args.output.write_text(render_text(ascii_frame), encoding="utf-8")
        return 0

    from asciify.io.save import sample_bg_fg

    peek_source = open_source(args.input, max_fps=args.max_fps, max_frames=1)
    first_frame = next(iter(peek_source))
    background, foreground = sample_bg_fg(first_frame, invert=args.invert)

    source = open_source(args.input, max_fps=args.max_fps, max_frames=args.max_frames)
    ascii_frames = (pipeline.process(f) for f in source)

    if args.output is None:
        import time

        HIDE_CURSOR = "\x1b[?25l"
        SHOW_CURSOR = "\x1b[?25h"
        CLEAR_SCREEN = "\x1b[2J\x1b[H"

        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()

        loops_done = 0
        try:
            while args.play is None or loops_done < args.play:
                for f in source:
                    af = pipeline.process(f)
                    text = (
                        render_terminal(af)
                        if args.color
                        else render_text(af)
                    )
                    sys.stdout.write(CLEAR_SCREEN)
                    sys.stdout.write(text + "\n")
                    sys.stdout.flush()
                    if af.duration_ms:
                        time.sleep(af.duration_ms / 1000.0)
                loops_done += 1
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write(SHOW_CURSOR)
            sys.stdout.flush()
        return 0

    from asciify.io.save import save_gif, save_video
    from asciify.ui.progress import ProgressUI

    out_suffix = args.output.suffix.lower()

    total_frames = _probe_frame_count(args.input, args.max_fps, args.max_frames)

    preview_rows = None if args.no_preview else args.preview_rows
    ui = ProgressUI(
        args.output,
        total_frames=total_frames,
        color=args.color,
        preview_rows=preview_rows,
    )

    try:
        if out_suffix == ".gif":
            save_gif(
                ascii_frames,
                args.output,
                font_size=args.font_size,
                background=background,
                foreground=foreground,
                total_frames=total_frames,
                on_progress=ui.on_frame,
            )
        elif out_suffix in _VIDEO_OUT_SUFFIXES:
            save_video(
                ascii_frames,
                args.output,
                fps=args.fps,
                font_size=args.font_size,
                background=background,
                foreground=foreground,
                crf=args.crf,
                preset=args.preset,
                total_frames=total_frames,
                on_progress=ui.on_frame,
            )
        else:
            lines: list[str] = []
            for i, af in enumerate(ascii_frames):
                lines.append(f"# Frame {i}")
                lines.append(render_text(af))
                ui.on_frame(i + 1, total_frames, af)
            args.output.write_text("\n".join(lines), encoding="utf-8")
    finally:
        ui.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
