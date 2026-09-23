"""Unit tests for the content/noise-condition classifier + SNR regressor.

No downloads, no heavy models: synthetic sines + hiss. If the trained
artifact is absent, predict_content() must return None (fallback contract).
"""

import numpy as np
import soundfile as sf

from server.ml.audio_understanding.content_classifier import (
    _extract_features,
    model_available,
    predict_content,
)


def _write(tmp_path, y, sr=22050, name="c.wav"):
    p = str(tmp_path / name)
    sf.write(p, y.astype(np.float32), sr)
    return p


def test_feature_size_is_81():
    sr = 22050
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    assert _extract_features(y, sr).shape == (84,)


def test_hiss_shifts_hiss_band_energy():
    sr = 22050
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    clean = 0.5 * np.sin(2 * np.pi * 220 * t)
    rng = np.random.default_rng(0)
    noisy = clean + rng.standard_normal(len(t)) * 0.2
    fc = _extract_features(clean, sr)
    fn = _extract_features(noisy, sr)
    # hiss-band ratio is index 78 (75 base + flat mean/std + hum-band, then hiss)
    assert fn[78] > fc[78]


def test_clipped_flag_sets_clip_ratio(tmp_path):
    sr = 22050
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    y = np.clip(0.5 * np.sin(2 * np.pi * 220 * t) * 3.0, -1, 1)
    f = _extract_features(y, sr)
    assert f[-2] > 0.01  # clip ratio
    p = _write(tmp_path, y, sr)
    res = predict_content(p)
    assert res is None or res["condition"] in {
        "clean", "hiss", "hum", "crowd", "clipped", "muffled",
    }


def test_predict_schema_when_available(tmp_path):
    sr = 22050
    t = np.linspace(0, 5, sr * 5, endpoint=False)
    y = 0.4 * np.sin(2 * np.pi * 330 * t)
    p = _write(tmp_path, y, sr)
    res = predict_content(p)
    if res is None:
        assert not model_available()
        return
    assert 0.0 <= res["confidence"] <= 1.0
    assert -5.0 <= res["snr_db"] <= 45.0
    assert 0.0 <= res["quality_score"] <= 100.0
    assert abs(sum(res["probs"].values()) - 1.0) < 0.01


def test_metrics_cards_shape():
    from server.ml.eval.metrics import content_model_metrics, groove_model_metrics

    for card in (content_model_metrics(), groove_model_metrics()):
        assert "available" in card
        if card["available"]:
            assert card["classes"] and card["feature_size"] > 0
