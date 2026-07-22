import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg


def extract_audio(video_path: str | Path, output_format: str = "wav") -> str:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    output_path = tempfile.mktemp(suffix=f".{output_format}")

    cmd = [
        ffmpeg_exe,
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le" if output_format == "wav" else "copy",
        "-ar", "44100",
        "-ac", "2",
        "-y",
        output_path,
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path
