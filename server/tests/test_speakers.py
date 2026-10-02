"""Speaker diarization + speaker-level editing.

Guards two regressions found by measurement:
  - VAD thresholded on `rms > percentile(rms, 40) * 1.5`, which excluded real
    speech whenever speech occupied most of the file (returned zero segments).
  - Cluster selection required `len(segments) >= k + 2`, so a 3-segment clip
    could never resolve 2 speakers.
"""

import numpy as np
import pytest
import soundfile as sf

from server.ml.diarization.diarization import _vad_segments, diarize
from server.ml.diarization.transcript import (
    chapters_from_transcript,
    complement_spans,
    speaker_spans,
    transcribe_with_speakers,
)

SR = 22050


def _voice(f0, dur, rng, amp=0.35):
    t = np.arange(int(dur * SR)) / SR
    sig = np.zeros_like(t)
    for h, a in ((1, 1.0), (2, 0.6), (3, 0.4), (4, 0.25), (5, 0.15)):
        sig += a * np.sin(2 * np.pi * f0 * h * t)
    sig *= 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    sig += 0.01 * rng.standard_normal(len(t))
    sig /= np.max(np.abs(sig)) + 1e-9
    return (amp * sig).astype(np.float32)


@pytest.fixture
def two_speaker(tmp_path):
    """A-B-A pattern: low voice, high voice, low voice again."""
    rng = np.random.default_rng(0)
    clip = np.concatenate([
        _voice(120, 3.0, rng),
        np.zeros(int(0.9 * SR), np.float32),
        _voice(210, 3.0, rng),
        np.zeros(int(0.9 * SR), np.float32),
        _voice(120, 2.5, rng),
    ])
    p = tmp_path / "two.wav"
    sf.write(str(p), clip, SR)
    return str(p), clip


@pytest.fixture
def single_speaker(tmp_path):
    rng = np.random.default_rng(3)
    clip = _voice(150, 6.0, rng)
    p = tmp_path / "one.wav"
    sf.write(str(p), clip, SR)
    return str(p), clip


class TestVad:
    def test_finds_all_turns(self, two_speaker):
        path, _ = two_speaker
        y, sr = __import__("librosa").load(path, sr=16000, mono=True)
        segs = _vad_segments(y, sr)
        assert len(segs) == 3, f"expected 3 speech turns, got {len(segs)}"

    def test_boundaries_land_on_the_gaps(self, two_speaker):
        """Turns are 0-3.0s, 3.9-6.9s, 7.8-10.3s with 0.9s gaps between."""
        path, _ = two_speaker
        y, sr = __import__("librosa").load(path, sr=16000, mono=True)
        segs = _vad_segments(y, sr)
        starts = [a / sr for a, _ in segs]
        ends = [b / sr for _, b in segs]
        assert starts[0] < 0.1, starts
        assert 2.9 < ends[0] < 3.2, ends
        assert 3.7 < starts[1] < 4.1, starts
        assert 6.8 < ends[1] < 7.1, ends
        assert 7.6 < starts[2] < 7.9, starts
        # gaps must land inside the 0.9s silences
        assert starts[1] - ends[0] > 0.7
        assert starts[2] - ends[1] > 0.7

    def test_silence_yields_nothing(self):
        y = np.zeros((16000 * 3,), dtype=np.float32)
        assert _vad_segments(y, 16000) == []


class TestDiarize:
    def test_finds_two_speakers(self, two_speaker):
        path, _ = two_speaker
        res = diarize(path)
        assert len(res["speakers"]) == 2, res["speakers"]

    def test_same_speaker_repeats_are_merged(self, two_speaker):
        """Turn 1 and turn 3 are the same voice and must share a label."""
        path, _ = two_speaker
        res = diarize(path)
        segs = res["segments"]
        assert len(segs) == 3
        assert segs[0]["speaker"] == segs[2]["speaker"], segs

    def test_single_speaker_stays_single(self, single_speaker):
        path, _ = single_speaker
        res = diarize(path)
        assert len(res["speakers"]) == 1, res["speakers"]


class TestSpans:
    def test_speaker_spans_partitions_correctly(self, two_speaker):
        path, _ = two_speaker
        t = transcribe_with_speakers(path, include_text=False)
        spk = t["speakers"][0]
        keep, drop = speaker_spans(t, {spk})
        assert keep and drop
        assert all(s["speaker"] == spk for s in t["segments"]
                   if any(a == s["start"] for a, _ in keep))

    def test_complement_is_exact_inverse(self):
        spans = [(1.0, 2.0), (4.0, 5.0)]
        comp = complement_spans(spans, duration=8.0)
        assert comp == [(0.0, 1.0), (2.0, 4.0), (5.0, 8.0)]

    def test_complement_of_nothing_is_empty(self):
        assert complement_spans([], 10.0) == []

    def test_talk_time_sums_to_segment_durations(self, two_speaker):
        path, _ = two_speaker
        t = transcribe_with_speakers(path, include_text=False)
        total = sum(s["duration"] for s in t["segments"])
        assert abs(sum(t["talk_time_seconds"].values()) - total) < 0.05


class TestChapters:
    def test_marks_real_pauses(self):
        t = {
            "duration_seconds": 30.0,
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "intro words", "speaker": "A"},
                {"start": 12.0, "end": 18.0, "text": "second part here", "speaker": "A"},
                {"start": 19.0, "end": 24.0, "text": "third", "speaker": "A"},
            ],
        }
        marks = chapters_from_transcript(t, min_gap=3.0)
        assert len(marks) == 2
        assert marks[1]["time"] == 12.0
        assert "second" in marks[1]["label"]
        assert marks[0]["time"] == 0.0

    def test_no_long_pauses_means_single_chapter(self):
        t = {
            "duration_seconds": 10.0,
            "segments": [{"start": 0.0, "end": 5.0, "text": "a"},
                         {"start": 5.2, "end": 9.0, "text": "b"}],
        }
        assert len(chapters_from_transcript(t, min_gap=3.0)) == 1

    def test_empty_transcript_is_safe(self):
        assert chapters_from_transcript({"segments": [], "duration_seconds": 5.0}) == []


class TestTranscriptIntegration:
    def test_includes_text_when_requested(self, two_speaker):
        path, _ = two_speaker
        t = transcribe_with_speakers(path, include_text=True)
        assert all("text" in s for s in t["segments"])