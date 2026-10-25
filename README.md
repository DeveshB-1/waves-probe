# waves-probe

A customer-side probe for Smallest Waves TTS. It streams a corpus of voice-agent lines through `/waves/v1/tts/live`, replays each chunk timeline through a simulated player to find the moments a caller would hear silence, round-trips the audio through Pulse STT to check that OTPs, amounts and names survive, and exits non-zero when an SLO breaks.

Stdlib only. Python 3.10+.

## Why this exists for Smallest

The cookbook ships Pulse STT benchmarks (`speech-to-text/benchmarks/`). There's no TTS counterpart.

And TTFB alone can't answer the question that matters on a call: did the caller hear a gap? A stream can land its first chunk in 90 ms and still stall for 400 ms mid-sentence. The Systems & Reliability JD puts it directly: "A healthy pod does not mean a successful call. If the caller hears silence, we investigate it as a failure." waves-probe measures that failure from the client side, where the caller is.

What it reports:

- **TTFB** and **time to audio** (TTFB plus whatever a jitter buffer waits for), as p50/p95/p99 over pooled raw samples. Per-run percentiles are never averaged.
- **Underruns**: each gap where the next chunk arrived after the player ran out of audio. Rate of streams affected, silence per stream, longest gap.
- **RTF**: wall time over audio time.
- **Round-trip intelligibility** (`--stt`): corpus WER plus a `must_hear` check per line, so "482916" has to come back as those 6 digits in order. Both sides go through the same normalizer, so `₹12,450` matches "rupees 12450" and `3:30 PM` matches "3.30 p.m.".
- **SLO gate**: `--slo slo.example.json` exits 1 on breach, so it can sit in CI before a model or infra rollout.

## Quickstart

```bash
# no key: simulated backend, exercises everything
python -m waves_probe --mock healthy --stt --slo slo.example.json
python -m waves_probe --mock degraded --stt --slo slo.example.json   # exits 1

# real API
export SMALLEST_API_KEY=sk_...
python -m waves_probe --model lightning_v3.1_pro --voice meher --runs 5 --concurrency 4 \
    --stt --slo slo.example.json --out results.json

# what a player with a 300 ms jitter buffer would experience
python -m waves_probe --jitter-buffer-ms 300 --runs 5
```

Corpus format (`corpus/voice_agent_en.jsonl`, one case per line):

```json
{"id": "otp", "text": "Your one time password is 482916. Please don't share it with anyone.", "must_hear": ["482916"]}
```

`reference` (what the caller should hear) defaults to `text`. `--stt-runs N` caps how many runs per case get transcribed, since STT costs credits.

SLO keys are `<metric>_max` or `<metric>_min`, with `.` into nested stats: `ttfb_ms.p95_max`, `underrun_stream_rate_max`, `must_hear_pass_rate_min`. A metric with no data (WER without `--stt`) is reported as skipped.

## Demo run (mock backend)

These numbers come from `waves_probe/mock.py`, a simulated server. They are **not** measurements of Smallest's API. I didn't have a key. The `degraded` profile keeps first-chunk latency close to `healthy` and adds occasional mid-stream stalls; the transcripts come from a hand-written fixture (`fixtures/mock_transcripts.json`) with one deliberate mishearing ("Iyer" as "Ayer").

```
$ python -m waves_probe --mock healthy --stt --runs 5 --slo slo.example.json
waves-probe  backend=mock:healthy  model=lightning_v3.1_pro  voice=meher  requests=50  concurrency=4  jitter_buffer=0 ms

| metric | p50 | p95 | p99 | max |
|---|---|---|---|---|
| TTFB | 87 ms | 112 ms | 134 ms | 134 ms |
| time to audio | 87 ms | 112 ms | 134 ms | 134 ms |
| RTF (wall/audio) | 0.202 | 0.22 | 0.232 | 0.232 |
| silence per stream | 0 ms | 0 ms | 0 ms | 0 ms |

streams with audible underrun: 0.0%   longest gap: 0 ms
errors: 0.0%
round-trip (TTS -> STT) on 10 streams: corpus WER 0.6%   must-hear pass 90.0%
  - doctor: did not hear 'doctor raghavendra iyer'

SLO: PASS
exit=0

$ python -m waves_probe --mock degraded --stt --runs 5 --slo slo.example.json
waves-probe  backend=mock:degraded  model=lightning_v3.1_pro  voice=meher  requests=50  concurrency=4  jitter_buffer=0 ms

| metric | p50 | p95 | p99 | max |
|---|---|---|---|---|
| TTFB | 98 ms | 131 ms | 143 ms | 143 ms |
| time to audio | 98 ms | 131 ms | 143 ms | 143 ms |
| RTF (wall/audio) | 0.336 | 0.665 | 0.755 | 0.755 |
| silence per stream | 0 ms | 147 ms | 401 ms | 401 ms |

streams with audible underrun: 6.0%   longest gap: 401 ms
errors: 0.0%
round-trip (TTS -> STT) on 10 streams: corpus WER 0.6%   must-hear pass 90.0%
  - doctor: did not hear 'doctor raghavendra iyer'

SLO: FAIL
  - underrun_stream_rate = 0.06 breaches max 0.02
exit=1
```

TTFB p95 moved from 112 ms to 131 ms, well inside a 250 ms target. A TTFB dashboard would stay green. 3 of 50 streams still had up to 401 ms of dead air, and the gate caught it.

## Tests

```bash
uv run --group dev pytest -q
# 36 passed in 0.59s
```

They cover the SSE parser (CRLF, comments, error frames, missing trailing blank line), the playback simulator, normalizer equivalences, pooled vs averaged percentiles, the SLO gate, determinism under concurrency, and the real `WavesTTS`/`PulseSTT` clients against a local HTTP server that speaks the Waves wire format.

## What's real, what isn't

- **Real**: the HTTP clients. The request shapes follow `smallest-inc/smallest-python-sdk` (SSE frames `data: {"audio": "<base64-pcm>"}`, closed by `data: {"done": true}`) and the Pulse batch script in `smallest-inc/cookbook`. Against `api.smallest.ai` with an invalid key, the probe gets `TTS HTTP 401: {"error":"unauthorized"}`, prints it and exits 2. A run with a valid key hasn't happened yet.
- **Simulated**: every number in the demo above.
- **Limits**: SSE only (WebSocket streaming is the next transport to add). Underrun detection assumes s16le mono PCM and a player that starts once its buffer fills. The normalizer is English-only and small; read `textnorm.py` before trusting a WER figure on other languages.

## Layout

```
waves_probe/sse.py        SSE frame parser for /tts/live
waves_probe/clients.py    WavesTTS (SSE) and PulseSTT clients, stdlib urllib
waves_probe/playback.py   jitter-buffer player simulation, underrun detection
waves_probe/textnorm.py   spoken-form normalizer, WER, must-hear check
waves_probe/runner.py     concurrent runs, aggregation, SLO gate
waves_probe/mock.py       simulated backend (healthy / degraded)
corpus/                   voice-agent test lines
```
