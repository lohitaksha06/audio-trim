import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

MODEL_NAME = "openai/whisper-tiny"


class Transcriber:
    """Whisper wrapper.

    Uses the `language`/`task` generation kwargs rather than the deprecated
    `forced_decoder_ids`, which this transformers build warns about.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        self._processor = WhisperProcessor.from_pretrained(self.model_name)
        self._model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def transcribe_array(
        self, y: np.ndarray, sr: int, language: str = "en"
    ) -> dict:
        """Transcribe an in-memory mono signal (already at any sample rate)."""
        self._load()
        mono = np.asarray(y, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=0)
        if mono.size < sr // 20:  # < 50 ms: nothing to transcribe
            return {
                "text": "",
                "duration_seconds": round(mono.size / sr, 3),
                "model": self.model_name,
            }
        if sr != 16000:
            mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
            sr = 16000

        input_features = self._processor(
            mono, sampling_rate=sr, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features.to(self.device),
                language=language,
                task="transcribe",
            )

        text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {
            "text": text.strip(),
            "duration_seconds": round(float(mono.size / sr), 3),
            "model": self.model_name,
        }

    def transcribe(self, audio_path: str) -> dict:
        self._load()
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        result = self.transcribe_array(y, sr)
        result["model"] = self.model_name
        return result