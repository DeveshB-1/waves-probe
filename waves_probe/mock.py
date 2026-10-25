"""Deterministic stand-ins for Waves TTS and Pulse STT.

These exist so the harness and its tests run without an API key. The
numbers they produce describe a simulated server, never Smallest's. The
`degraded` profile keeps first-chunk latency close to `healthy` and adds
mid-stream stalls, which is the failure a TTFB-only dashboard misses.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .clients import StreamTimeline, TTSRequest

PROFILES = {
    # ttfb (mean, sd) ms | ms of compute per 100 ms of audio | stall prob per stream | stall ms range
    "healthy": dict(ttfb=(85, 15), gen_per_100ms=(18, 4), stall_p=0.0, stall_ms=(0, 0)),
    "degraded": dict(ttfb=(95, 20), gen_per_100ms=(30, 12), stall_p=0.35, stall_ms=(250, 900)),
}

MS_PER_CHAR = 62.0  # rough speaking rate used to size fake audio
CHUNK_AUDIO_MS = 80.0


@dataclass
class MockTTS:
    profile: str = "healthy"
    seed: int = 7

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"unknown mock profile {self.profile!r}; pick from {sorted(PROFILES)}")

    def stream(self, req: TTSRequest) -> StreamTimeline:
        p = PROFILES[self.profile]
        # seeded per (case, run) so results do not depend on thread scheduling
        rng = random.Random(f"{self.seed}:{req.case_id}:{req.run}:{req.text}")
        audio_ms = max(CHUNK_AUDIO_MS, len(req.text) * MS_PER_CHAR / req.speed)
        n = max(1, round(audio_ms / CHUNK_AUDIO_MS))
        bytes_per_chunk = int(req.sample_rate * 2 * CHUNK_AUDIO_MS / 1000)
        stall_at = rng.randrange(2, n) if n > 3 and rng.random() < p["stall_p"] else -1

        t = max(20.0, rng.gauss(*p["ttfb"]))
        chunks = []
        for i in range(n):
            if i:
                t += max(2.0, rng.gauss(*p["gen_per_100ms"]) * CHUNK_AUDIO_MS / 100)
            if i == stall_at:
                t += rng.uniform(*p["stall_ms"])
            chunks.append((t, bytes_per_chunk))
        return StreamTimeline(chunks=chunks, pcm=b"\x00\x00" * (bytes_per_chunk // 2) * n, total_ms=t + 5)


class MockSTT:
    """Returns a canned transcript per corpus case id (fixtures/mock_transcripts.json)."""

    def __init__(self, fixture: Path):
        self.transcripts: dict[str, str] = json.loads(Path(fixture).read_text())

    def transcribe(self, wav: bytes, language: str, case_id: str = "") -> str:
        if case_id not in self.transcripts:
            raise KeyError(f"no mock transcript for case {case_id!r}")
        return self.transcripts[case_id]
