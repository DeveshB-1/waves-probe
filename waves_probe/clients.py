"""Real HTTP clients for Waves TTS (SSE) and Pulse STT. Stdlib only.

Endpoints follow smallest-inc/smallest-python-sdk and smallest-inc/cookbook:
    POST https://api.smallest.ai/waves/v1/tts/live      (SSE, base64 PCM chunks)
    POST https://api.smallest.ai/waves/v1/stt/?model=&language=   (raw audio body)
"""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlencode

from .sse import iter_audio

API_BASE = os.environ.get("SMALLEST_API_BASE", "https://api.smallest.ai/waves/v1")


@dataclass
class TTSRequest:
    text: str
    voice_id: str = "meher"
    model: str = "lightning_v3.1_pro"
    sample_rate: int = 24000
    speed: float = 1.0
    case_id: str = ""  # bookkeeping for the mock; never sent upstream
    run: int = 0

    def payload(self) -> dict:
        return {
            "text": self.text,
            "voice_id": self.voice_id,
            "model": self.model,
            "sample_rate": self.sample_rate,
            "speed": self.speed,
            "output_format": "pcm",
        }


@dataclass
class StreamTimeline:
    """What the client saw: (arrival_ms, nbytes) per chunk plus the PCM."""

    chunks: list[tuple[float, int]] = field(default_factory=list)
    pcm: bytes = b""
    total_ms: float = 0.0


class TTSClient(Protocol):
    def stream(self, req: TTSRequest) -> StreamTimeline: ...


class STTClient(Protocol):
    def transcribe(self, wav: bytes, language: str, case_id: str = "") -> str: ...


class APIError(RuntimeError):
    pass


def wav_from_pcm(pcm: bytes, sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, channels,
        sample_rate, byte_rate, channels * bits // 8, bits, b"data", len(pcm),
    )
    return header + pcm


def _key(api_key: str | None) -> str:
    key = api_key or os.environ.get("SMALLEST_API_KEY")
    if not key:
        raise APIError("SMALLEST_API_KEY is not set (use --mock to run without one)")
    return key


class WavesTTS:
    def __init__(self, api_key: str | None = None, base: str = API_BASE, timeout: float = 60):
        self.key, self.base, self.timeout = _key(api_key), base.rstrip("/"), timeout

    def stream(self, req: TTSRequest) -> StreamTimeline:
        http_req = urllib.request.Request(
            f"{self.base}/tts/live",
            data=json.dumps(req.payload()).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        tl = StreamTimeline()
        parts: list[bytes] = []
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(http_req, timeout=self.timeout) as resp:
                for pcm in iter_audio(resp):  # HTTPResponse yields lines as they arrive
                    tl.chunks.append(((time.perf_counter() - t0) * 1000, len(pcm)))
                    parts.append(pcm)
        except urllib.error.HTTPError as e:
            raise APIError(f"TTS HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}") from e
        tl.total_ms = (time.perf_counter() - t0) * 1000
        tl.pcm = b"".join(parts)
        return tl


class PulseSTT:
    def __init__(self, api_key: str | None = None, base: str = API_BASE, model: str = "pulse", timeout: float = 120):
        self.key, self.base, self.model, self.timeout = _key(api_key), base.rstrip("/"), model, timeout

    def transcribe(self, wav: bytes, language: str, case_id: str = "") -> str:
        url = f"{self.base}/stt/?{urlencode({'model': self.model, 'language': language})}"
        http_req = urllib.request.Request(
            url, data=wav, method="POST",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(http_req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise APIError(f"STT HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}") from e
        text = body.get("transcription")
        if not isinstance(text, str):
            raise APIError(f"STT response missing 'transcription': {str(body)[:200]}")
        return text
