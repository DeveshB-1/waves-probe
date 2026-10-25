"""Simulate a client playback buffer over a chunk arrival timeline.

TTFB says when the first bytes landed. It says nothing about whether the
caller heard a gap halfway through a sentence. This module replays the
arrival timeline against a real-time player and reports every moment the
player ran dry (an underrun = silence the caller hears).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    arrival_ms: float  # time since the request was sent
    audio_ms: float  # playable duration of this chunk


@dataclass
class PlaybackResult:
    time_to_audio_ms: float  # when the player actually starts speaking
    audio_ms: float
    underruns: list[float] = field(default_factory=list)  # each gap in ms

    @property
    def silence_ms(self) -> float:
        return sum(self.underruns)

    @property
    def max_gap_ms(self) -> float:
        return max(self.underruns, default=0.0)


def pcm_ms(nbytes: int, sample_rate: int, bytes_per_sample: int = 2, channels: int = 1) -> float:
    return nbytes / (sample_rate * bytes_per_sample * channels) * 1000.0


def simulate(chunks: list[Chunk], jitter_buffer_ms: float = 0.0, min_gap_ms: float = 20.0) -> PlaybackResult:
    """Replay `chunks` (in arrival order) through a player.

    The player waits until `jitter_buffer_ms` of audio is buffered (or the
    stream ends), then plays in real time. A gap shorter than `min_gap_ms`
    is ignored: most players and codecs smooth that over.
    """
    if not chunks:
        raise ValueError("no audio chunks")
    total = sum(c.audio_ms for c in chunks)

    buffered = 0.0
    start_idx = len(chunks) - 1
    for i, c in enumerate(chunks):
        buffered += c.audio_ms
        if buffered >= jitter_buffer_ms:
            start_idx = i
            break

    start = chunks[start_idx].arrival_ms
    play_end = start + sum(c.audio_ms for c in chunks[: start_idx + 1])
    gaps: list[float] = []
    for c in chunks[start_idx + 1 :]:
        if c.arrival_ms > play_end:
            gap = c.arrival_ms - play_end
            if gap >= min_gap_ms:
                gaps.append(gap)
            play_end = c.arrival_ms + c.audio_ms
        else:
            play_end += c.audio_ms
    return PlaybackResult(time_to_audio_ms=start, audio_ms=total, underruns=gaps)
