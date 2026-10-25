"""Run a corpus through TTS (and optionally STT), aggregate, check SLOs."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import playback, textnorm
from .clients import STTClient, TTSClient, TTSRequest, wav_from_pcm
from .stats import summarize


@dataclass
class Case:
    id: str
    text: str
    reference: str = ""  # what the caller should hear; defaults to text
    must_hear: list[str] = field(default_factory=list)
    language: str = "en"
    tags: list[str] = field(default_factory=list)


@dataclass
class Record:
    case_id: str
    run: int
    ok: bool
    error: str = ""
    ttfb_ms: float = math.nan
    time_to_audio_ms: float = math.nan
    total_ms: float = math.nan
    audio_ms: float = math.nan
    rtf: float = math.nan  # wall time / audio time; < 1 means faster than real time
    chunks: int = 0
    underruns: list[float] = field(default_factory=list)
    transcript: str | None = None
    wer_edits: int = 0
    wer_words: int = 0
    missing: list[str] = field(default_factory=list)


def load_corpus(path: Path) -> list[Case]:
    cases = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("#"):
            try:
                cases.append(Case(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"{path}:{n}: bad case: {e}") from e
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate case ids")
    return cases


def probe_one(tts: TTSClient, stt: STTClient | None, case: Case, run: int, *, voice: str, model: str,
              sample_rate: int, jitter_buffer_ms: float) -> Record:
    rec = Record(case_id=case.id, run=run, ok=False)
    try:
        tl = tts.stream(TTSRequest(case.text, voice, model, sample_rate, case_id=case.id, run=run))
        if not tl.chunks:
            raise RuntimeError("stream finished with no audio")
        chunks = [playback.Chunk(t, playback.pcm_ms(n, sample_rate)) for t, n in tl.chunks]
        pb = playback.simulate(chunks, jitter_buffer_ms=jitter_buffer_ms)
        rec.ttfb_ms, rec.time_to_audio_ms, rec.total_ms = tl.chunks[0][0], pb.time_to_audio_ms, tl.total_ms
        rec.audio_ms, rec.chunks, rec.underruns = pb.audio_ms, len(chunks), pb.underruns
        rec.rtf = tl.total_ms / pb.audio_ms
        if stt is not None:
            rec.transcript = stt.transcribe(wav_from_pcm(tl.pcm, sample_rate), case.language, case.id)
            rec.wer_edits, rec.wer_words = textnorm.wer(case.reference or case.text, rec.transcript)
            rec.missing = textnorm.missing_phrases(case.must_hear, rec.transcript)
        rec.ok = True
    except Exception as e:  # one failed request is a data point, not a crash
        rec.error = f"{type(e).__name__}: {e}"
    return rec


def run(tts: TTSClient, stt: STTClient | None, cases: list[Case], *, runs: int = 3, concurrency: int = 4,
        voice: str = "meher", model: str = "lightning_v3.1_pro", sample_rate: int = 24000,
        jitter_buffer_ms: float = 0.0, stt_runs: int = 1) -> list[Record]:
    """`stt_runs` caps how many runs per case are transcribed (STT costs credits)."""
    jobs = [(c, r) for r in range(runs) for c in cases]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futs = [pool.submit(probe_one, tts, stt if r < stt_runs else None, c, r, voice=voice, model=model,
                            sample_rate=sample_rate, jitter_buffer_ms=jitter_buffer_ms) for c, r in jobs]
        return [f.result() for f in futs]


def aggregate(records: list[Record]) -> dict:
    ok = [r for r in records if r.ok]
    scored = [r for r in ok if r.transcript is not None]
    words = sum(r.wer_words for r in scored)
    return {
        "requests": len(records),
        "error_rate": (len(records) - len(ok)) / len(records) if records else math.nan,
        "ttfb_ms": summarize([r.ttfb_ms for r in ok]),
        "time_to_audio_ms": summarize([r.time_to_audio_ms for r in ok]),
        "rtf": summarize([r.rtf for r in ok]),
        "underrun_stream_rate": sum(1 for r in ok if r.underruns) / len(ok) if ok else math.nan,
        "max_gap_ms": max((g for r in ok for g in r.underruns), default=0.0),
        "silence_ms_per_stream": summarize([sum(r.underruns) for r in ok]),
        "scored": len(scored),
        "corpus_wer": sum(r.wer_edits for r in scored) / words if words else math.nan,
        "must_hear_pass_rate": sum(1 for r in scored if not r.missing) / len(scored) if scored else math.nan,
        "failures": sorted({(r.case_id, p) for r in scored for p in r.missing}),
    }


# SLO keys: "<metric path>_<max|min>". Metric path uses "." for nested stats, e.g. ttfb_ms.p95_max.
def check_slos(summary: dict, slos: dict[str, float]) -> tuple[list[str], list[str]]:
    """Return (breaches, skipped). A metric with no data (e.g. WER without --stt) is skipped."""
    breaches, skipped = [], []
    for key, limit in slos.items():
        path, _, bound = key.rpartition("_")
        if bound not in ("max", "min") or not path:
            raise ValueError(f"SLO key {key!r} must end in _max or _min")
        value = summary
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"SLO key {key!r}: no metric {path!r}")
            value = value[part]
        if isinstance(value, float) and math.isnan(value):
            skipped.append(path)
        elif (bound == "max" and value > limit) or (bound == "min" and value < limit):
            breaches.append(f"{path} = {value:.4g} breaches {bound} {limit}")
    return breaches, skipped


def to_json(records: list[Record], summary: dict, breaches: list[str], meta: dict) -> str:
    return json.dumps({"meta": meta, "summary": summary, "slo_breaches": breaches,
                       "records": [asdict(r) for r in records]}, indent=2, default=str)
