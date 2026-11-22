"""Real-prompt parsing: polish, timed adds, chord shorthand, slow fades."""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from server.ml.audio_operations import _add_instrument, execute_plan
from server.ml.prompt_engine import Intent, regex_plan_from_prompt


def test_timed_add_targets_the_added_instrument():
    plan = regex_plan_from_prompt(
        "add drums from 30 seconds at the same level as the backing, below the voice"
    )
    assert plan.intent == Intent.ADD_INSTRUMENT
    assert plan.params["instrument"] == "drums"
    assert plan.params["start_seconds"] == 30.0


def test_polish_prompt_carries_the_chain():
    plan = regex_plan_from_prompt(
        "clean up the recording, add a little reverb, adjust the volume and reduce background noise"
    )
    assert plan.intent == Intent.POLISH
    assert plan.params["denoise"] is True
    assert plan.params["reverb_mix"] == pytest.approx(0.12)
    assert plan.params["normalize"] is True


def test_perfect_means_cleanup_and_level_not_reverb():
    plan = regex_plan_from_prompt("make my music perfect")
    assert plan.intent == Intent.POLISH
    assert plan.params["reverb_mix"] == 0.0
    assert plan.params["normalize"] is True


def test_slash_chord_shorthand():
    plan = regex_plan_from_prompt("use a C/G")
    assert plan.intent == Intent.ADD_INSTRUMENT
    assert plan.params["chords"] == ["C/G"]


def test_slow_fade_gets_a_longer_default():
    plan = regex_plan_from_prompt("fade out slowly at the end")
    assert plan.intent == Intent.FADE
    assert plan.params["fade_out_dur"] == 4.0


def test_colon_time_becomes_start_seconds():
    plan = regex_plan_from_prompt("add drums from 0:30")
    assert plan.intent == Intent.ADD_INSTRUMENT
    assert plan.params["start_seconds"] == 30.0


def test_added_layer_is_silent_before_its_start():
    sr = 22050
    t = np.linspace(0, 4, sr * 4, endpoint=False)
    y = np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2)
    mixed, meta, layer = _add_instrument(y, sr, "drums", {"start_seconds": 2.0})
    assert meta["hits"] > 0
    layer2 = layer if layer.ndim == 2 else layer[np.newaxis, :]
    before = np.abs(layer2[:, : int(1.99 * sr)]).max()
    after = np.abs(layer2[:, int(2.2 * sr):]).max()
    assert before < 1e-6
    assert after > 0


def test_polish_executes_and_reports_steps():
    sr = 22050
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    src = os.path.join(tempfile.gettempdir(), "polish_input.wav")
    sf.write(src, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)

    plan = regex_plan_from_prompt("make my music perfect")
    result = execute_plan(src, plan)
    assert result.get("output_path")
    assert result["metadata"]["method"] == "deterministic_polish"
    assert result["metadata"]["polish_steps"]
    y, _ = sf.read(result["output_path"])
    assert float(np.abs(y).max()) <= 0.95
