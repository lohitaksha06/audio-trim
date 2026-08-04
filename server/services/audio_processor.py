import tempfile
from pathlib import Path

import librosa
import numpy as np


def analyze_audio(audio_path: str | Path) -> dict:
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_indices = np.sum(chroma, axis=1)
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key = keys[int(np.argmax(key_indices))]

    rms = librosa.feature.rms(y=y)[0]
    energy_score = float(np.mean(rms))

    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "duration_seconds": round(duration, 1),
        "bpm": round(tempo, 1),
        "key": f"{key} (estimated)",
        "energy": round(energy_score, 4),
        "sample_rate": sr,
        "channels": 1,
    }


def remove_silence(audio_path: str | Path) -> str:
    y, sr = librosa.load(str(audio_path), sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    output_path = tempfile.mktemp(suffix=".wav")
    import soundfile as sf
    sf.write(output_path, y_trimmed, sr)
    return output_path
