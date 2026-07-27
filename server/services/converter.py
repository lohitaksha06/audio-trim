import os
import tempfile
from pathlib import Path

import imageio_ffmpeg
import soundfile as sf
from pydub import AudioSegment

pydub_audio_segment = AudioSegment

FORMAT_MAP = {
    "mp3": "mp3",
    "wav": "wav",
    "flac": "flac",
    "aac": "adts",
    "ogg": "ogg",
    "m4a": "mp4",
}

SUPPORTED_FORMATS = set(FORMAT_MAP.keys())


def _get_ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ensure_ffmpeg():
    ff = _get_ffmpeg()
    AudioSegment.converter = ff
    AudioSegment.ffmpeg = ff
    AudioSegment.ffprobe = ff.replace("ffmpeg", "ffprobe")


def convert_file(input_path: str, target_format: str) -> str:
    if target_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported target format: {target_format}. Supported: {SUPPORTED_FORMATS}")

    _ensure_ffmpeg()
    ext = Path(input_path).suffix.lower().lstrip(".")

    if ext == target_format:
        return input_path

    out_dir = Path(tempfile.gettempdir()) / "audelle-converted"
    out_dir.mkdir(exist_ok=True)

    output_path = str(out_dir / f"{Path(input_path).stem}.{target_format}")

    if ext == "wav" or target_format == "wav":
        y, sr = sf.read(input_path)
        sf.write(output_path if target_format == "wav" else input_path, y, sr)

    if target_format == "wav" and ext != "wav":
        audio = AudioSegment.from_file(input_path, format=ext)
        audio.export(output_path, format="wav")
        return output_path

    audio = AudioSegment.from_file(input_path, format=ext if ext != "wav" else "wav")
    audio.export(output_path, format=target_format)
    return output_path
