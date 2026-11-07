"""Download and instantiate the model weights once.

Run this during a one-off deploy step, not on every request. It keeps the
first real separation/transcription request from paying a multi-minute
HuggingFace/Demucs download and makes the cache directory explicit.

    python scripts/warm_models.py

Respects TORCH_HOME and HF_HOME; mount those on a persistent volume in
production.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.ml.source_separation.separator import SourceSeparator  # noqa: E402
from server.ml.transcription.transcriber import Transcriber  # noqa: E402


def main() -> None:
    for model in ("htdemucs_ft", "htdemucs_6s"):
        print(f"warming Demucs {model} …", flush=True)
        separator = SourceSeparator(model)
        separator._load()
        print(f"  sources: {separator.sources}", flush=True)

    print("warming Whisper tiny …", flush=True)
    transcriber = Transcriber()
    transcriber._load()
    print("  whisper ready", flush=True)


if __name__ == "__main__":
    main()
