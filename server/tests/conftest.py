import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tone_path(tmp_path_factory):
    """Generate a 10s two-tone WAV once for the whole test session."""
    path = tmp_path_factory.mktemp("audio") / "tone.wav"
    sr = 22050
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)
    sf.write(str(path), y, sr)
    return str(path)


@pytest.fixture(scope="session")
def speech_path(tmp_path_factory):
    """A speech-like two-tone signal for VAD/diarization tests."""
    path = tmp_path_factory.mktemp("audio") / "speech.wav"
    sr = 16000

    def tone(freq, dur, amp):
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        return amp * (np.sin(2 * np.pi * freq * t) + 0.4 * np.sin(2 * np.pi * 2 * freq * t))

    signal = np.concatenate(
        [
            tone(220, 2.0, 0.6),
            tone(880, 1.5, 0.3),
            np.zeros(int(sr * 0.8)),
            tone(300, 2.0, 0.6),
            tone(700, 1.5, 0.35),
        ]
    )
    sf.write(str(path), signal, sr)
    return str(path)