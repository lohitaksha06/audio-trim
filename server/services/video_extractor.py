import subprocess
from pathlib import Path

import imageio_ffmpeg

from server.utils.tempfile_util import temp_output_path


def extract_audio(video_path: str | Path, output_format: str = "wav") -> str:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    output_path = temp_output_path(f".{output_format}")

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


def mux_audio_onto_video(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Write ``audio_path`` back into ``video_path``, keeping the video stream.

    This closes the "audio to video muxing" item that was listed as unconfirmed:
    only extraction had ever been implemented, so a user who uploaded an .mp4
    edited its audio and then had no way to get a video back out.

    Design notes:
    * The video stream is **stream-copied** (``-c:v copy``), not re-encoded.
      Audelle is an audio tool; re-encoding video would cost minutes per clip
      and lose quality for no reason.
    * ``-shortest`` truncates to the shorter input so a short edit does not
      desynchronise the result. ``apad`` was tried first to pad a short edit up
      to the source duration, but ``apad`` combined with ``-c:v copy`` makes
      ffmpeg **hang indefinitely** (measured: no output after 45 s, while
      ``-shortest`` alone finished in 0.2 s). Because the video is stream-copied
      its duration is already fixed by the container, so padding is unnecessary.
    * The original audio stream is dropped (``-map 0:v:0``), since it is the
      thing being replaced.

    Returns the output path.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    out = str(output_path) if output_path is not None else temp_output_path(".mp4")

    cmd = [
        ffmpeg_exe,
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        "-y",
        out,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip().splitlines()
        tail = detail[-1] if detail else "unknown ffmpeg error"
        raise RuntimeError(f"Muxing failed: {tail}") from e
    return out


def normalize_to_wav(audio_path: str | Path) -> str:
    """Decode a compressed audio file to WAV so librosa can read it.

    libsndfile (soundfile, which librosa prefers) reads WAV/FLAC/OGG/MP3 but has
    no AAC decoder, and this box has no ffmpeg, avconv or gstreamer on PATH --
    ``audioread.available_backends()`` returns only ``RawAudioFile``. So an
    .m4a/.aac upload made ``librosa.load`` raise inside ``analyze_audio``, the
    whole upload returned 500, and the UI reported "upload a file first" even
    though the filename was already on screen.

    Uses the ffmpeg binary bundled with imageio-ffmpeg, the same one
    ``extract_audio`` already uses for video.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    output_path = temp_output_path(".wav")
    cmd = [
        ffmpeg_exe,
        "-i", str(audio_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-y",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path
