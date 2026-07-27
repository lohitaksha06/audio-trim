import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


class Transcriber:
    def __init__(self, model_name: str = "openai/whisper-tiny", device: str | None = None):
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
        self._model.config.forced_decoder_ids = self._processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )

    def transcribe(self, audio_path: str) -> dict:
        self._load()

        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        input_features = self._processor(
            y, sampling_rate=16000, return_tensors="pt"
        ).input_features

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features.to(self.device),
                return_timestamps=True,
            )

        transcription = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        duration = float(len(y)) / 16000

        return {
            "text": transcription.strip(),
            "duration_seconds": duration,
            "model": self.model_name,
        }
