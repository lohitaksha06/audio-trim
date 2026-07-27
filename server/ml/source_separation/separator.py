import tempfile
from pathlib import Path
from typing import Dict

import torch


class SourceSeparator:
    def __init__(self, model_name: str = "htdemucs", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self._separator = None

    def _load(self):
        if self._separator is not None:
            return
        from demucs.separate import Separator

        self._separator = Separator(
            model=self.model_name,
            device=self.device,
            shifts=1,
            overlap=0.25,
            split=True,
            progress=False,
        )

    @property
    def sources(self) -> list[str]:
        self._load()
        return list(self._separator.model.sources)

    def separate(self, audio_path: str, out_dir: str | None = None) -> Dict[str, str]:
        self._load()
        if out_dir is None:
            out_dir = tempfile.mkdtemp(prefix="audelle_stems_")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        origin, sources = self._separator.separate_audio_file(Path(audio_path))

        from demucs.audio import save_audio

        result = {}
        for name, source in sources.items():
            stem_path = out_path / f"{name}.wav"
            save_audio(
                source.cpu(),
                str(stem_path),
                samplerate=self._separator.samplerate,
            )
            result[name] = str(stem_path)

        return result

    def isolate(self, audio_path: str, target: str, out_dir: str | None = None) -> str | None:
        stems = self.separate(audio_path, out_dir)
        return stems.get(target)
