"""Parser for the Waves `/waves/v1/tts/live` Server-Sent Events stream.

Contract (from the smallest-python-sdk docstring for the SSE route):
    event: audio
    data: {"audio": "<base64-pcm>"}

    data: {"done": true}
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass
class SSEFrame:
    event: str
    data: dict


class SSEError(RuntimeError):
    pass


def iter_frames(lines: Iterable[bytes | str]) -> Iterator[SSEFrame]:
    """Yield one frame per blank-line-terminated SSE block.

    Multiple `data:` lines in one block are joined with newlines, as the SSE
    spec says. Comment lines (starting with ':') are ignored.
    """
    event = "message"
    data_lines: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if line == "":
            if data_lines:
                yield SSEFrame(event, _load("\n".join(data_lines)))
            event, data_lines = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:  # stream closed without a trailing blank line
        yield SSEFrame(event, _load("\n".join(data_lines)))


def _load(text: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SSEError(f"non-JSON SSE data: {text[:120]!r}") from exc
    if not isinstance(obj, dict):
        raise SSEError(f"SSE data is not an object: {text[:120]!r}")
    return obj


def iter_audio(lines: Iterable[bytes | str]) -> Iterator[bytes]:
    """Yield decoded PCM chunks until the `done` frame. Raises on error frames."""
    for frame in iter_frames(lines):
        if frame.data.get("done"):
            return
        if "error" in frame.data or frame.event == "error":
            raise SSEError(f"server error frame: {frame.data}")
        audio = frame.data.get("audio")
        if audio:
            yield base64.b64decode(audio)
