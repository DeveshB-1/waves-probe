from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import runner
from .clients import PulseSTT, WavesTTS
from .mock import PROFILES, MockSTT, MockTTS

ROOT = Path(__file__).resolve().parent.parent


def _fmt(x: float, unit: str = "") -> str:
    return "n/a" if isinstance(x, float) and math.isnan(x) else f"{x:,.0f}{unit}" if unit == " ms" else f"{x:.3g}{unit}"


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.1%}"


def render(summary: dict, breaches: list[str], meta: dict) -> str:
    s = summary
    lines = [
        f"waves-probe  backend={meta['backend']}  model={meta['model']}  voice={meta['voice']}  "
        f"requests={s['requests']}  concurrency={meta['concurrency']}  jitter_buffer={meta['jitter_buffer_ms']:.0f} ms",
        "",
        "| metric | p50 | p95 | p99 | max |",
        "|---|---|---|---|---|",
    ]
    for name, key, unit in (("TTFB", "ttfb_ms", " ms"), ("time to audio", "time_to_audio_ms", " ms"),
                            ("RTF (wall/audio)", "rtf", ""), ("silence per stream", "silence_ms_per_stream", " ms")):
        d = s[key]
        lines.append(f"| {name} | {_fmt(d['p50'], unit)} | {_fmt(d['p95'], unit)} | {_fmt(d['p99'], unit)} | {_fmt(d['max'], unit)} |")
    lines += [
        "",
        f"streams with audible underrun: {_pct(s['underrun_stream_rate'])}   longest gap: {s['max_gap_ms']:,.0f} ms",
        f"errors: {s['error_rate']:.1%}",
    ]
    if s["scored"]:
        lines.append(f"round-trip (TTS -> STT) on {s['scored']} streams: corpus WER {s['corpus_wer']:.1%}   "
                     f"must-hear pass {s['must_hear_pass_rate']:.1%}")
        for case_id, phrase in s["failures"]:
            lines.append(f"  - {case_id}: did not hear {phrase!r}")
    lines.append("")
    if meta.get("slo"):
        lines.append("SLO: PASS" if not breaches else "SLO: FAIL\n" + "\n".join(f"  - {b}" for b in breaches))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="waves-probe", description="Customer-side latency and intelligibility probe for Smallest Waves TTS.")
    ap.add_argument("--corpus", type=Path, default=ROOT / "corpus" / "voice_agent_en.jsonl")
    ap.add_argument("--model", default="lightning_v3.1_pro")
    ap.add_argument("--voice", default="meher")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--jitter-buffer-ms", type=float, default=0.0, help="audio the player buffers before speaking")
    ap.add_argument("--stt", action="store_true", help="round-trip audio through Pulse STT and score intelligibility")
    ap.add_argument("--stt-model", default="pulse")
    ap.add_argument("--stt-runs", type=int, default=1, help="transcribe only the first N runs of each case")
    ap.add_argument("--mock", choices=sorted(PROFILES), help="simulated backend, no API key needed")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--slo", type=Path, help="JSON of SLO limits; exit 1 on breach")
    ap.add_argument("--out", type=Path, help="write per-request records + summary JSON here")
    a = ap.parse_args(argv)

    cases = runner.load_corpus(a.corpus)
    if a.mock:
        tts, stt = MockTTS(a.mock, a.seed), (MockSTT(ROOT / "fixtures" / "mock_transcripts.json") if a.stt else None)
    else:
        tts, stt = WavesTTS(), (PulseSTT(model=a.stt_model) if a.stt else None)

    records = runner.run(tts, stt, cases, runs=a.runs, concurrency=a.concurrency, voice=a.voice, model=a.model,
                         sample_rate=a.sample_rate, jitter_buffer_ms=a.jitter_buffer_ms, stt_runs=a.stt_runs)
    summary = runner.aggregate(records)
    breaches, skipped = runner.check_slos(summary, json.loads(a.slo.read_text())) if a.slo else ([], [])
    meta = {"backend": f"mock:{a.mock}" if a.mock else "api.smallest.ai", "model": a.model, "voice": a.voice,
            "concurrency": a.concurrency, "jitter_buffer_ms": a.jitter_buffer_ms, "corpus": str(a.corpus), "slo": str(a.slo) if a.slo else None}
    print(render(summary, breaches, meta))
    if skipped:
        print(f"SLO skipped (no data): {', '.join(skipped)}")
    if a.out:
        a.out.write_text(runner.to_json(records, summary, breaches, meta))
    errors = sorted({r.error for r in records if not r.ok})
    for e in errors[:3]:
        print(f"error: {e}", file=sys.stderr)
    if len(errors) and summary["error_rate"] == 1.0:
        return 2
    return 1 if breaches else 0


if __name__ == "__main__":
    sys.exit(main())
