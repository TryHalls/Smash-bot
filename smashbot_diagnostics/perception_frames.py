"""Streaming exact offline frames from Task 008 H.264 captures.

The decoder is FFmpeg, while frame identity comes only from the existing
Task 008 ``packets.json`` metadata.  No NumPy/OpenCV or derived video files
are required.
"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class FrameStreamError(RuntimeError):
    """Raised when FFmpeg output cannot satisfy the exact frame contract."""


@dataclass(frozen=True)
class FrameMetadata:
    source_run: str
    frame_index: int
    pts_us: int
    width: int
    height: int
    pixel_format: str = "rgb24"


@dataclass(frozen=True)
class OfflineFrame:
    source_run: str
    frame_index: int
    pts_us: int
    width: int
    height: int
    pixel_format: str
    pixels: bytes

    @property
    def byte_size(self) -> int:
        return len(self.pixels)


def load_frame_metadata(
    packets_json: Path,
    *,
    source_run: str,
    width: int,
    height: int,
    pixel_format: str = "rgb24",
) -> list[FrameMetadata]:
    """Load exact media-frame index/PTS pairs from Task 008 metadata."""

    import json

    with Path(packets_json).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    result: list[FrameMetadata] = []
    if width <= 0 or height <= 0:
        raise FrameStreamError("frame dimensions must be positive")
    _pixel_bytes_per_frame(width, height, pixel_format)
    seen: set[int] = set()
    for packet in document.get("media_packets", []):
        if packet.get("is_config"):
            continue
        frame_index = packet.get("media_frame_index")
        pts_us = packet.get("scrcpy_pts_us")
        if not isinstance(frame_index, int) or not isinstance(pts_us, int):
            raise FrameStreamError("media packet metadata lacks integer frame index/PTS")
        if frame_index in seen:
            raise FrameStreamError(f"duplicate media frame index: {frame_index}")
        seen.add(frame_index)
        result.append(FrameMetadata(source_run, frame_index, pts_us, width, height, pixel_format))
    result.sort(key=lambda item: item.frame_index)
    if not result:
        raise FrameStreamError("packets.json contains no media frames")
    if any(current.pts_us <= previous.pts_us for previous, current in zip(result, result[1:])):
        raise FrameStreamError("media PTS must be strictly increasing by frame index")
    return result


def _pixel_bytes_per_frame(width: int, height: int, pixel_format: str) -> int:
    if width <= 0 or height <= 0:
        raise FrameStreamError("frame dimensions must be positive")
    if pixel_format == "rgb24":
        channels = 3
    elif pixel_format == "gray":
        channels = 1
    else:
        raise FrameStreamError(f"unsupported raw pixel format: {pixel_format}")
    return width * height * channels


class FFmpegFrameStream:
    """Bounded-memory FFmpeg frame stream with exact identity checks.

    ``iter_sequential`` yields every metadata frame.  ``iter_selected`` and
    ``iter_range`` request only exact frame indices through FFmpeg's select
    filter; the output order is checked against the requested metadata.
    Consumers should use this object as a context manager so early iteration
    stops terminate the child process deterministically.
    """

    def __init__(
        self,
        source_h264: Path,
        metadata: Iterable[FrameMetadata],
        *,
        ffmpeg: str = "ffmpeg",
        pixel_format: str = "rgb24",
    ):
        self.source_h264 = Path(source_h264)
        self.metadata = sorted(list(metadata), key=lambda item: item.frame_index)
        self.ffmpeg = ffmpeg
        self.pixel_format = pixel_format
        if not self.metadata:
            raise FrameStreamError("frame stream requires metadata")
        if any(item.pixel_format != pixel_format for item in self.metadata):
            raise FrameStreamError("metadata pixel format does not match stream pixel format")
        if len({item.frame_index for item in self.metadata}) != len(self.metadata):
            raise FrameStreamError("frame metadata contains duplicate indices")
        if any(item.frame_index < 0 for item in self.metadata):
            raise FrameStreamError("frame metadata contains a negative frame index")
        if any(current.pts_us <= previous.pts_us for previous, current in zip(self.metadata, self.metadata[1:])):
            raise FrameStreamError("frame metadata PTS must be strictly increasing")
        if not self.source_h264.is_file():
            raise FrameStreamError(f"H.264 source does not exist: {self.source_h264}")
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_tail = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._stdout_drain_thread: threading.Thread | None = None
        self._extra_stdout = threading.Event()

    def __enter__(self) -> "FFmpegFrameStream":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for thread in (self._stderr_thread, self._stdout_drain_thread):
            if thread is not None:
                thread.join(timeout=1)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._stderr_thread = None
        self._stdout_drain_thread = None
        self._process = None

    def _resolve_indices(self, selected: Iterable[int] | None) -> list[FrameMetadata]:
        by_index = {item.frame_index: item for item in self.metadata}
        if selected is None:
            return list(self.metadata)
        indices = list(selected)
        if not indices:
            raise FrameStreamError("selected frame indices cannot be empty")
        if indices != sorted(set(indices)):
            raise FrameStreamError("selected frame indices must be sorted and unique")
        missing = [index for index in indices if index not in by_index]
        if missing:
            raise FrameStreamError(f"requested frame indices are absent from metadata: {missing[:5]}")
        return [by_index[index] for index in indices]

    def _command(self, selected: list[FrameMetadata] | None) -> list[str]:
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-i",
            str(self.source_h264),
        ]
        if selected is not None:
            expression = "+".join(f"eq(n\\,{item.frame_index})" for item in selected)
            command.extend(["-vf", f"select={expression}", "-vsync", "0", "-frames:v", str(len(selected))])
        command.extend(["-f", "rawvideo", "-pix_fmt", self.pixel_format, "pipe:1"])
        return command

    @staticmethod
    def _read_exact(stream: Any, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _start_stderr_drain(self, stream: Any) -> None:
        self._stderr_tail = bytearray()

        def drain() -> None:
            try:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        return
                    self._stderr_tail.extend(chunk)
                    if len(self._stderr_tail) > 64 * 1024:
                        del self._stderr_tail[: len(self._stderr_tail) - 64 * 1024]
            except (OSError, ValueError):
                return

        self._stderr_thread = threading.Thread(target=drain, name="task009-ffmpeg-stderr", daemon=True)
        self._stderr_thread.start()

    def _start_stdout_drain(self, stream: Any) -> None:
        self._extra_stdout.clear()

        def drain() -> None:
            try:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        return
                    self._extra_stdout.set()
            except (OSError, ValueError):
                return

        self._stdout_drain_thread = threading.Thread(target=drain, name="task009-ffmpeg-extra-stdout", daemon=True)
        self._stdout_drain_thread.start()

    def _iter(self, selected: list[FrameMetadata] | None) -> Iterator[OfflineFrame]:
        expected = self._resolve_indices(selected)
        frame_size = _pixel_bytes_per_frame(expected[0].width, expected[0].height, self.pixel_format)
        if any(_pixel_bytes_per_frame(item.width, item.height, self.pixel_format) != frame_size for item in expected):
            raise FrameStreamError("selected frames do not have a stable byte size")
        process: subprocess.Popen[bytes] | None = None
        try:
            try:
                process = subprocess.Popen(
                    self._command(expected if selected is not None else None),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except OSError as exc:
                raise FrameStreamError(f"failed to start FFmpeg: {exc}") from exc
            self._process = process
            assert process.stdout is not None
            if process.stderr is not None:
                self._start_stderr_drain(process.stderr)
            for item in expected:
                payload = self._read_exact(process.stdout, frame_size)
                if len(payload) != frame_size:
                    raise FrameStreamError(
                        f"short raw frame for index {item.frame_index}: {len(payload)} != {frame_size}"
                    )
                yield OfflineFrame(
                    item.source_run,
                    item.frame_index,
                    item.pts_us,
                    item.width,
                    item.height,
                    item.pixel_format,
                    payload,
                )
            self._start_stdout_drain(process.stdout)
            try:
                return_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise FrameStreamError("FFmpeg did not terminate after expected frames") from exc
            if self._stdout_drain_thread is not None:
                self._stdout_drain_thread.join(timeout=1)
            if self._extra_stdout.is_set():
                raise FrameStreamError("FFmpeg emitted an extra raw frame")
            if return_code != 0:
                stderr = bytes(self._stderr_tail).decode("utf-8", errors="replace")
                raise FrameStreamError(f"FFmpeg exited with {return_code}: {stderr[-1000:]}")
        finally:
            if process is not None:
                self.close()

    def iter_sequential(self) -> Iterator[OfflineFrame]:
        return self._iter(None)

    def iter_selected(self, indices: Iterable[int]) -> Iterator[OfflineFrame]:
        selected = list(indices)
        self._resolve_indices(selected)
        return self._iter(selected)

    def iter_range(self, first_frame_index: int, last_frame_index: int) -> Iterator[OfflineFrame]:
        if last_frame_index < first_frame_index:
            raise FrameStreamError("frame range must be ascending")
        return self.iter_selected(range(first_frame_index, last_frame_index + 1))
