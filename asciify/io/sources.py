from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np
from PIL import Image

from asciify.core.frame import Frame


class FrameSource(Protocol):
    def __iter__(self) -> Iterator[Frame]: ...  # eh idk what it did but hopefully it doesnt break


class ImageFrameSource:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def __iter__(self) -> Iterator[Frame]:
        with Image.open(self._path) as img:
            pixels = np.asarray(img.convert("RGB"), dtype=np.uint8)
        yield Frame(pixels=pixels, index=0)


class GifFrameSource:
    DEFAULT_DURATION_MS: float = 100.0

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def __iter__(self) -> Iterator[Frame]:
        with Image.open(self._path) as gif:
            if not hasattr(gif, "n_frames"):
                pixels = np.asarray(gif.convert("RGB"), dtype=np.uint8)
                yield Frame(pixels=pixels, index=0, duration_ms=None)
                return

            for idx in range(gif.n_frames):
                gif.seek(idx)
                pixels = np.asarray(gif.convert("RGB"), dtype=np.uint8)

                duration_ms = gif.info.get("duration", self.DEFAULT_DURATION_MS)
                yield Frame(pixels=pixels, index=idx, duration_ms=float(duration_ms))


class VideoFrameSource:
    def __init__(
        self,
        path: str | Path,
        *,
        max_fps: float | None = None,
        max_frames: int | None = None,
        stream_index: int = 0,
    ) -> None:
        self._path = Path(path)
        self._max_fps = max_fps
        self._max_frames = max_frames
        self._stream_index = stream_index

    def __iter__(self) -> Iterator[Frame]:
        try:
            import av
        except ImportError as exc:
            raise ImportError(
                "VideoFrameSource requires PyAV. Install it with:  pip install av"
            ) from exc

        with av.open(str(self._path)) as container:
            stream = container.streams.video[self._stream_index]
            stream.thread_type = "AUTO"

            native_fps: float | None = None
            if stream.average_rate:
                native_fps = float(stream.average_rate)

            min_gap_pts: float | None = None
            if self._max_fps and native_fps and native_fps > self._max_fps:
                time_base = float(stream.time_base) if stream.time_base else 1.0
                min_gap_pts = (1.0 / self._max_fps) / time_base

            last_pts: float | None = None
            yielded = 0

            for packet in container.demux(stream):
                for av_frame in packet.decode():
                    if av_frame is None:
                        continue

                    if min_gap_pts is not None and av_frame.pts is not None:
                        if (
                            last_pts is not None
                            and (av_frame.pts - last_pts) < min_gap_pts
                        ):
                            continue
                        last_pts = av_frame.pts

                    duration_ms: float | None = None
                    if av_frame.pts is not None and stream.time_base:
                        tb = float(stream.time_base)
                        dur = av_frame.duration or 1
                        duration_ms = dur * tb * 1000.0

                    img = av_frame.to_image()
                    pixels = np.asarray(img.convert("RGB"), dtype=np.uint8)
                    yield Frame(pixels=pixels, index=yielded, duration_ms=duration_ms)

                    yielded += 1
                    if self._max_frames is not None and yielded >= self._max_frames:
                        return


_GIF_SUFFIXES = {".gif"}
_VIDEO_SUFFIXES = {
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


def open_source(
    path: str | Path,
    *,
    max_fps: float | None = None,
    max_frames: int | None = None,
) -> FrameSource:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in _GIF_SUFFIXES:
        return GifFrameSource(p)
    if suffix in _VIDEO_SUFFIXES:
        return VideoFrameSource(p, max_fps=max_fps, max_frames=max_frames)
    return ImageFrameSource(p)
