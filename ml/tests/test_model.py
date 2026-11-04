from pathlib import Path

import pytest

from instruction_ml.data import load_jsonl
from instruction_ml.model import BaselineModel

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "seed_prompts.jsonl"


@pytest.fixture(scope="module")
def model():
    rows = load_jsonl(SEEDS)
    return BaselineModel().fit(rows)


def test_model_roundtrip(model, tmp_path):
    path = tmp_path / "baseline.joblib"
    model.save(path)
    loaded = BaselineModel.load(path)
    prediction = loaded.predict("remove the vocals")
    assert prediction["ops"] == ["remove_stem"]
    assert "vocals" in prediction["targets"]


def test_model_predicts_an_operation(model):
    for prompt in [
        "make the drums louder by 6 db",
        "clean up the background noise",
        "fade out the ending",
    ]:
        prediction = model.predict(prompt)
        assert prediction["ops"], prompt


def test_model_amount_is_bounded(model):
    prediction = model.predict("boost the bass by 4 db")
    assert prediction["amount"] is None or -24 <= prediction["amount"] <= 24
