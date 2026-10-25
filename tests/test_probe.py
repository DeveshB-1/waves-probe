import base64
import io
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from waves_probe import cli, playback, runner, sse, stats, textnorm
from waves_probe.clients import PulseSTT, TTSRequest, WavesTTS, wav_from_pcm
from waves_probe.mock import MockSTT, MockTTS

ROOT = Path(__file__).resolve().parent.parent


# ---------- SSE parsing (format from the smallest-python-sdk SSE docstring) ----------

def _frames(*chunks: bytes, done=True, trailing_blank=True) -> list[bytes]:
    out = []
    for c in chunks:
        out += [b"event: audio\n", b"data: " + json.dumps({"audio": base64.b64encode(c).decode()}).encode() + b"\n", b"\n"]
    if done:
        out += [b'data: {"done": true}\n'] + ([b"\n"] if trailing_blank else [])
    return out


def test_sse_decodes_audio_until_done():
    lines = _frames(b"\x01\x02", b"\x03\x04") + _frames(b"\xff\xff", done=False)
    assert list(sse.iter_audio(lines)) == [b"\x01\x02", b"\x03\x04"]


def test_sse_handles_crlf_comments_and_missing_trailing_blank():
    lines = [b": keepalive\r\n", b"\r\n"] + [line.replace(b"\n", b"\r\n") for line in _frames(b"ab", trailing_blank=False)]
    assert list(sse.iter_audio(lines)) == [b"ab"]


def test_sse_error_frame_raises():
    with pytest.raises(sse.SSEError):
        list(sse.iter_audio([b'data: {"error": "quota exceeded"}\n', b"\n"]))


def test_sse_non_json_raises():
    with pytest.raises(sse.SSEError):
        list(sse.iter_audio([b"data: not json\n", b"\n"]))


# ---------- playback simulation ----------

def C(t, a=80.0):
    return playback.Chunk(t, a)


def test_no_underrun_when_chunks_arrive_ahead_of_playhead():
    r = playback.simulate([C(100), C(120), C(140), C(160)])
    assert r.time_to_audio_ms == 100 and r.underruns == [] and r.audio_ms == 320


def test_mid_stream_stall_is_reported_as_silence():
    # playhead after 2 chunks ends at 100 + 160 = 260; chunk 3 arrives at 700 -> 440 ms of silence
    r = playback.simulate([C(100), C(150), C(700), C(710)])
    assert r.underruns == [440] and r.max_gap_ms == 440


def test_jitter_buffer_delays_start_and_absorbs_a_short_stall():
    chunks = [C(100), C(150), C(200), C(450), C(460)]
    assert playback.simulate(chunks).underruns == [110]  # 100+240=340 -> 450
    buffered = playback.simulate(chunks, jitter_buffer_ms=240)
    assert buffered.time_to_audio_ms == 200 and buffered.underruns == []  # 200+240=440 -> 450 gap 10 < 20


def test_tiny_gaps_below_threshold_are_ignored():
    assert playback.simulate([C(100), C(190)], min_gap_ms=20).underruns == []


def test_pcm_ms():
    assert playback.pcm_ms(48000, 24000) == 1000.0


# ---------- text normalization / scoring ----------

@pytest.mark.parametrize("n,words", [(0, "zero"), (15, "fifteen"), (40, "forty"), (105, "one hundred five"),
                                     (12450, "twelve thousand four hundred fifty"), (2_000_001, "two million one")])
def test_int_to_words(n, words):
    assert textnorm.int_to_words(n) == words


@pytest.mark.parametrize("a,b", [
    ("Your OTP is 482916.", "your otp is 4 8 2 9 1 6"),
    ("EMI of ₹12,450", "emi of rupees 12450"),
    ("10.5% interest", "ten point five percent interest"),
    ("at 3:30 PM", "at 3.30 p.m."),
    ("due on 5 November", "due on 5th november"),
    ("ref K7QX9", "ref k 7 q x 9"),
    ("support@example.com", "support at example dot com"),
    ("Dr. Iyer", "doctor iyer"),
])
def test_equivalent_spoken_forms_normalize_equal(a, b):
    assert textnorm.normalize(a) == textnorm.normalize(b)


def test_wer_counts_edits_against_reference_length():
    assert textnorm.wer("connecting you to doctor iyer", "connecting you to doctor ayer") == (1, 5)
    assert textnorm.wer("a b c", "a b c") == (0, 3)


def test_missing_phrases_needs_contiguous_match():
    hyp = "your otp is 4 8 2 9 1 6 please"
    assert textnorm.missing_phrases(["482916"], hyp) == []
    assert textnorm.missing_phrases(["482961"], hyp) == ["482961"]


# ---------- percentiles: pool samples, never average p95s ----------

def test_percentile_nearest_rank():
    xs = list(range(1, 101))
    assert stats.percentile(xs, 50) == 50 and stats.percentile(xs, 95) == 95 and stats.percentile(xs, 100) == 100
    assert math.isnan(stats.percentile([], 95))


def test_pooled_p95_differs_from_mean_of_run_p95s():
    run_a, run_b = [100.0] * 20, [100.0] * 19 + [2000.0]
    mean_of_p95 = (stats.percentile(run_a, 95) + stats.percentile(run_b, 95)) / 2
    assert mean_of_p95 == 100.0
    assert stats.percentile(run_a + run_b, 99) == 2000.0  # the tail only shows up when pooled


# ---------- runner + SLO gate on the mock backend ----------

def _run(profile, **kw):
    cases = runner.load_corpus(ROOT / "corpus" / "voice_agent_en.jsonl")
    stt = MockSTT(ROOT / "fixtures" / "mock_transcripts.json")
    return runner.run(MockTTS(profile), stt, cases, **kw)


def test_mock_runs_are_deterministic_under_concurrency():
    a = [(r.case_id, r.run, r.ttfb_ms, r.underruns) for r in _run("degraded", runs=3, concurrency=8)]
    b = [(r.case_id, r.run, r.ttfb_ms, r.underruns) for r in _run("degraded", runs=3, concurrency=1)]
    assert a == b


def test_degraded_profile_fails_underrun_slo_while_ttfb_slo_passes():
    s = runner.aggregate(_run("degraded", runs=5))
    slos = json.loads((ROOT / "slo.example.json").read_text())
    breaches, skipped = runner.check_slos(s, slos)
    assert skipped == []
    assert any(b.startswith("underrun_stream_rate") for b in breaches)
    assert not any(b.startswith("ttfb_ms") for b in breaches)


def test_healthy_profile_passes_and_flags_the_misheard_name():
    s = runner.aggregate(_run("healthy", runs=2))
    breaches, _ = runner.check_slos(s, json.loads((ROOT / "slo.example.json").read_text()))
    assert breaches == []
    assert s["failures"] == [("doctor", "doctor raghavendra iyer")]
    assert s["scored"] == 10  # stt_runs=1: only the first run of each case is transcribed


def test_slo_without_stt_skips_intelligibility_metrics():
    cases = runner.load_corpus(ROOT / "corpus" / "voice_agent_en.jsonl")
    s = runner.aggregate(runner.run(MockTTS("healthy"), None, cases, runs=1))
    breaches, skipped = runner.check_slos(s, {"corpus_wer_max": 0.1, "ttfb_ms.p95_max": 10})
    assert skipped == ["corpus_wer"] and breaches[0].startswith("ttfb_ms.p95")


def test_bad_slo_key_is_rejected():
    with pytest.raises(ValueError):
        runner.check_slos({"rtf": {}}, {"rtf.p95": 1})


def test_failed_request_is_recorded_not_raised():
    class Boom:
        def stream(self, req):
            raise ConnectionResetError("peer reset")
    recs = runner.run(Boom(), None, [runner.Case("x", "hello")], runs=2)
    s = runner.aggregate(recs)
    assert s["error_rate"] == 1.0 and "peer reset" in recs[0].error


def test_cli_exit_codes(capsys, tmp_path):
    slo = str(ROOT / "slo.example.json")
    assert cli.main(["--mock", "healthy", "--stt", "--runs", "2", "--slo", slo]) == 0
    out = tmp_path / "r.json"
    assert cli.main(["--mock", "degraded", "--stt", "--runs", "5", "--slo", slo, "--out", str(out)]) == 1
    assert "SLO: FAIL" in capsys.readouterr().out
    assert json.loads(out.read_text().replace("NaN", "null"))["slo_breaches"]


# ---------- real HTTP clients against a local fake of the Waves API ----------

class _FakeWaves(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _FakeWaves.seen.append((self.path, self.headers.get("Authorization"), body))
        if self.path == "/waves/v1/tts/live":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for line in _frames(b"\x00\x01" * 2400, b"\x00\x01" * 2400):
                self.wfile.write(line)
                self.wfile.flush()
        elif self.path.startswith("/waves/v1/stt/"):
            out = json.dumps({"transcription": "hello there", "metadata": {"duration": 0.2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        else:
            self.send_response(401)
            self.end_headers()


@pytest.fixture
def fake_api():
    srv = HTTPServer(("127.0.0.1", 0), _FakeWaves)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _FakeWaves.seen = []
    yield f"http://127.0.0.1:{srv.server_address[1]}/waves/v1"
    srv.shutdown()


def test_real_clients_speak_the_waves_wire_format(fake_api):
    tts = WavesTTS(api_key="sk_test", base=fake_api)
    tl = tts.stream(TTSRequest("hello there", case_id="never-sent"))
    assert len(tl.chunks) == 2 and len(tl.pcm) == 2 * 4800
    path, auth, body = _FakeWaves.seen[0]
    sent = json.loads(body)
    assert auth == "Bearer sk_test" and sent["output_format"] == "pcm" and "case_id" not in sent and "run" not in sent

    stt = PulseSTT(api_key="sk_test", base=fake_api)
    assert stt.transcribe(wav_from_pcm(tl.pcm, 24000), "en") == "hello there"
    path, _, body = _FakeWaves.seen[1]
    assert path == "/waves/v1/stt/?model=pulse&language=en" and body[:4] == b"RIFF"


def test_missing_key_message(monkeypatch):
    monkeypatch.delenv("SMALLEST_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="--mock"):
        WavesTTS()
