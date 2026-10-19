"""RED: combine layers, boost quiet stems, base->bass, swing groove, strings."""

import os

import numpy as np
import pytest
import soundfile as sf

from server.ml.prompt_engine import Intent, extract_instrument, regex_plan_from_prompt
from server.ml.audio_operations import execute_plan


def test_base_guitar_is_bass():
    assert extract_instrument("now add a base guitar in the back too") == "bass"
    plan = regex_plan_from_prompt("now add a base guitar in the back too")
    assert plan.intent == Intent.ADD_INSTRUMENT
    assert plan.params["instrument"] == "bass"


def test_based_does_not_hijack():
    # "based on the groove" must not invent a bass instrument
    assert extract_instrument("add drums based on the groove") == "drums"


def test_combine_intents():
    for p in [
        "can u comebin both drum and base",
        "combine both drums and bass",
        "add both drums and bass",
        "layer drums and bass together",
        "merge drums, bass and guitar",
    ]:
        plan = regex_plan_from_prompt(p)
        assert plan.intent == Intent.COMBINE, f"{p} -> {plan.intent}"
    plan = regex_plan_from_prompt("combine both drums and bass")
    assert set(plan.params["instruments"]) >= {"drums", "bass"}


def test_single_add_not_combine():
    plan = regex_plan_from_prompt("can you add drums in the background")
    assert plan.intent == Intent.ADD_INSTRUMENT


def test_boost_intent():
    for p in ["i cant hear the drums", "the bass is too quiet", "make the drums louder", "boost the vocals"]:
        plan = regex_plan_from_prompt(p)
        assert plan.intent == Intent.BOOST, f"{p} -> {plan.intent}"
    assert regex_plan_from_prompt("i cant hear the drums").params["target"] == "drums"


def test_swing_groove_parses():
    plan = regex_plan_from_prompt("add swing drums")
    assert plan.intent == Intent.ADD_INSTRUMENT
    assert plan.params["groove"] == "swing"


def test_strings_executes(tone_path):
    plan = regex_plan_from_prompt("add strings")
    res = execute_plan(tone_path, plan)
    assert res["intent"] == "add_instrument"
    assert res.get("output_path")


def _band_energy(path, fmin, fmax):
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(np.abs(S[band]) ** 2))


def test_combine_executes_both(tone_path):
    plan = regex_plan_from_prompt("combine drums and bass")
    res = execute_plan(tone_path, plan)
    assert res["intent"] == "combine"
    assert res.get("output_path")
    assert set(res["metadata"]["combined"]) >= {"drums", "bass"}
    out = res["output_path"]
    # drums: kick-band energy up vs input
    assert _band_energy(out, 0, 150) > _band_energy(tone_path, 0, 150) * 1.5


def test_boost_drums_raises_kick_band(tone_path):
    plan = regex_plan_from_prompt("i cant hear the drums")
    res = execute_plan(tone_path, plan)
    assert res["intent"] == "boost"
    assert res.get("output_path")
    assert _band_energy(res["output_path"], 0, 200) > _band_energy(tone_path, 0, 200) * 1.2


def test_chained_tempo_stays_musical(tone_path):
    """Two stacked layers must not double the grid each time."""
    from server.ml.audio_operations import _add_instrument
    import librosa

    plan = regex_plan_from_prompt("add drums")
    r1 = execute_plan(tone_path, plan)
    y, sr = librosa.load(r1["output_path"], sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    y2, meta2, _ = _add_instrument(y, sr, "bass", {"groove": "default"})
    assert meta2["tempo_bpm"] <= 180, meta2


class TestCompressedAudioUpload:
    """.m4a/.aac must upload, not 500.

    libsndfile has no AAC decoder and this machine has no ffmpeg/avconv/
    gstreamer on PATH (audioread offers only RawAudioFile), so librosa.load
    raised NoBackendError inside analyze_audio. The upload returned 500 while
    the browser had already drawn the filename, so the UI said "upload a file
    first" about a file that was sitting right there.
    """

    @staticmethod
    def _make_m4a(tmpdir):
        import os
        import subprocess
        import tempfile

        import imageio_ffmpeg

        sr = 22050
        t = np.linspace(0, 3, sr * 3, endpoint=False)
        wav = os.path.join(tmpdir, "src.wav")
        sf.write(wav, 0.4 * np.sin(2 * np.pi * 220 * t), sr)
        m4a = os.path.join(tmpdir, "clip.m4a")
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-i", wav, "-c:a", "aac", "-b:a", "96k", m4a],
            check=True,
            capture_output=True,
        )
        return m4a

    def test_m4a_uploads_and_is_decoded_to_wav(self, tmp_path):
        from fastapi.testclient import TestClient
        from server.main import app

        m4a = self._make_m4a(str(tmp_path))
        # sanity: librosa genuinely cannot read it, which is why we transcode
        import librosa
        with pytest.raises(Exception):
            librosa.load(m4a, sr=None, mono=True)

        c = TestClient(app)
        up = c.post("/api/upload", files={"file": ("clip.m4a", open(m4a, "rb"), "audio/mp4")})
        assert up.status_code == 200, up.text[:300]
        body = up.json()
        assert body["audio_path"].endswith(".wav")
        assert body["analysis"]["duration_seconds"] > 0

    def test_m4a_upload_is_processable(self, tmp_path):
        from fastapi.testclient import TestClient
        from server.main import app

        m4a = self._make_m4a(str(tmp_path))
        c = TestClient(app)
        up = c.post("/api/upload", files={"file": ("clip.m4a", open(m4a, "rb"), "audio/mp4")})
        assert up.status_code == 200, up.text[:300]
        r = c.post("/api/process", json={"audio_path": up.json()["audio_path"], "prompt": "make it louder"})
        assert r.status_code == 200, r.text[:300]
        assert r.json()["intent"] != "unknown"

    def test_wav_upload_still_works(self, tmp_path):
        import os

        sr = 22050
        t = np.linspace(0, 3, sr * 3, endpoint=False)
        p = str(tmp_path / "plain.wav")
        sf.write(p, 0.4 * np.sin(2 * np.pi * 220 * t), sr)
        from fastapi.testclient import TestClient
        from server.main import app

        c = TestClient(app)
        up = c.post("/api/upload", files={"file": ("plain.wav", open(p, "rb"), "audio/wav")})
        assert up.status_code == 200, up.text[:300]
        assert os.path.exists(up.json()["audio_path"])


class TestVideoMuxing:
    """Edited audio must be written back onto the source video.

    Muxing was listed as "unconfirmed" because only extraction had ever been
    implemented. These tests build a real video, so they fail loudly if the
    ffmpeg invocation is wrong rather than silently passing.
    """

    @staticmethod
    def _make_video(path, seconds=2):
        import subprocess

        import imageio_ffmpeg

        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y",
                "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=15:duration={seconds}",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
            ],
            check=True,
            capture_output=True,
        )
        return str(path)

    def test_mux_keeps_video_and_replaces_audio(self, tmp_path):
        import subprocess

        import imageio_ffmpeg
        import librosa

        from server.services.video_extractor import extract_audio, mux_audio_onto_video

        vid = self._make_video(tmp_path / "in.mp4")
        sr = 22050
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        # 880 Hz, deliberately not the 440 Hz in the source video.
        edited = str(tmp_path / "edited.wav")
        sf.write(edited, (0.4 * np.sin(2 * np.pi * 880 * t)).astype(np.float32), sr)

        out = mux_audio_onto_video(vid, edited, str(tmp_path / "out.mp4"))
        assert os.path.exists(out) and os.path.getsize(out) > 0

        probe = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-i", out], capture_output=True, text=True
        ).stderr
        assert "Video:" in probe, "video stream was lost"
        assert "Audio:" in probe, "audio stream missing"

        y, _ = librosa.load(extract_audio(out), sr=22050, mono=True)
        spec = np.abs(np.fft.rfft(y))
        peak = float(np.fft.rfftfreq(len(y), 1 / 22050)[np.argmax(spec)])
        assert abs(peak - 880) < 40, f"audio not replaced (peak {peak:.0f} Hz)"

    def test_mux_does_not_hang(self, tmp_path):
        """`apad` + `-c:v copy` makes ffmpeg hang forever.

        A plain call that never returns fails the suite, so this asserts a
        bounded runtime instead of trusting the command shape.
        """
        import subprocess
        import time

        from server.services.video_extractor import mux_audio_onto_video

        vid = self._make_video(tmp_path / "in.mp4")
        sr = 22050
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        edited = str(tmp_path / "e.wav")
        sf.write(edited, (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32), sr)

        started = time.time()
        mux_audio_onto_video(vid, edited, str(tmp_path / "o.mp4"))
        assert time.time() - started < 30, "muxing exceeded 30s - likely hung"

    def test_process_returns_video_download_key(self, tmp_path):
        from fastapi.testclient import TestClient
        from server.main import app

        vid = self._make_video(tmp_path / "in.mp4")
        c = TestClient(app)
        up = c.post("/api/upload", files={"file": ("clip.mp4", open(vid, "rb"), "video/mp4")})
        assert up.status_code == 200, up.text[:300]
        body = up.json()
        assert body["is_video"] is True
        assert body["source_video_path"], "video uploads must expose the original for muxing"

        r = c.post(
            "/api/process",
            json={
                "audio_path": body["audio_path"],
                "prompt": "make it louder",
                "source_video_path": body["source_video_path"],
            },
        )
        assert r.status_code == 200, r.text[:300]
        assert r.json()["video_download_key"], "expected a muxed video download key"


def test_process_accepts_storage_key():
    """Chaining: a previous download_key must work as the next audio_path."""
    from fastapi.testclient import TestClient
    from server.main import app

    c = TestClient(app)
    import tempfile, os
    sr = 22050
    t = np.linspace(0, 4, sr * 4, endpoint=False)
    p = os.path.join(tempfile.gettempdir(), "chain_src.wav")
    sf.write(p, 0.4 * np.sin(2 * np.pi * 220 * t), sr)
    up = c.post("/api/upload", files={"file": ("c.wav", open(p, "rb"), "audio/wav")})
    assert up.status_code == 200
    ap = up.json()["audio_path"]
    r1 = c.post("/api/process", json={"audio_path": ap, "prompt": "add drums"})
    assert r1.status_code == 200
    key = r1.json()["download_key"]
    assert key
    # feed the storage key back in (what the frontend chains with)
    r2 = c.post("/api/process", json={"audio_path": key, "prompt": "add bass"})
    assert r2.status_code == 200, r2.text[:300]
    assert r2.json()["intent"] == "add_instrument"
