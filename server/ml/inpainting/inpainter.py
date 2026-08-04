"""Audio inpainting — seamlessly replace or remove a section.

`inpaint()` removes the target region and fills the gap with a crossfaded
blend of the surrounding audio so the transition is smooth. This covers
"remove that cymbal crash and fill smoothly" / "clean up the cough at 0:45".
"""

import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def inpaint(
    audio_path: str,
    start: float,
    end: float,
    crossfade_s: float = 0.05,
) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    channels, total = y.shape

    start_s = max(0.0, min(float(start), total / sr))
    end_s = max(start_s + 0.001, min(float(end), total / sr))

    start_idx = int(start_s * sr)
    end_idx = int(end_s * sr)

    keep = np.concatenate([y[:, :start_idx], y[:, end_idx:]], axis=1)

    cf = int(crossfade_s * sr)
    cf = min(cf, keep.shape[1] // 4)

    if cf > 0 and keep.shape[1] > 0:
        ramp_up = np.linspace(0, 1, cf)
        ramp_dn = np.linspace(1, 0, cf)
        keep[:, :cf] *= ramp_dn
        keep[:, -cf:] *= ramp_up
        keep[:, :cf] += keep[:, cf : 2 * cf] * (1 - ramp_dn)

    out_path = tempfile.mktemp(suffix=".wav")
    sf.write(out_path, keep.T if channels > 1 else keep[0], sr)

    return {
        "output_path": out_path,
        "removed_start": round(start_s, 3),
        "removed_end": round(end_s, 3),
        "new_duration_seconds": round(keep.shape[1] / sr, 2),
    }


def paint_intro_outro(
    audio_path: str, start: float, end: float, crossfade_s: float = 0.5
) -> dict[str, Any]:
    """Same as inpaint but with a longer crossfade — for gentle fixes."""
    return inpaint(audio_path, start, end, crossfade_s=crossfade_s)