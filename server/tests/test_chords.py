"""Chord-name parsing and the strummed chord layer."""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from server.ml.chords import (
    chord_midi,
    parse_chord_token,
    parse_progression,
    render_chord_progression,
)
from server.ml.prompt_engine import Intent, regex_plan_from_prompt


def test_parse_simple_and_extended_chords():
    assert parse_chord_token("C").name == "C"
    assert parse_chord_token("Am").root == 9
    assert parse_chord_token("Am").quality == "min"
    assert parse_chord_token("F#m").root == 6
    assert parse_chord_token("Bbmaj7").quality == "maj7"
    assert parse_chord_token("G7").quality == "dom7"
    assert parse_chord_token("Dsus4").quality == "sus4"
    assert parse_chord_token("C/G").bass == 7
    assert parse_chord_token("C/G").name == "C/G"


def test_rejects_non_chords():
    assert parse_chord_token("bass") is None
    assert parse_chord_token("hello") is None


def test_parse_progression_from_prompts():
    assert [c.name for c in parse_progression("add guitar chords: Am F C G")] == ["Am", "F", "C", "G"]
    assert len(parse_progression("play Am-F-C-G on guitar")) == 4
    assert len(parse_progression("strum Cmaj7 then Am7 then Fmaj7 then G7")) == 4
    assert parse_progression("add a bass") is None
    assert parse_progression("make it louder") is None
    assert parse_progression("play in C") is None


def test_chord_midi_notes():
    assert chord_midi(parse_chord_token("C")) == [48, 52, 55]
    assert chord_midi(parse_chord_token("Am")) == [57, 60, 64]


def test_render_progression_is_audible():
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    chords = [parse_chord_token(n) for n in ("C", "G", "Am", "F")]
    layer, count = render_chord_progression(chords, 120.0, beats, 4.0, 22050)
    assert count == 2
    assert layer.shape[0] == int(4.0 * 22050)
    assert float(np.abs(layer).max()) > 0.001


def test_prompt_routes_chord_request_to_add_instrument():
    plan = regex_plan_from_prompt("add guitar chords: Am F C G")
    assert plan.intent == Intent.ADD_INSTRUMENT
    assert plan.params["instrument"] == "guitar"
    assert plan.params["chords"] == ["Am", "F", "C", "G"]


def test_execute_adds_the_named_chords():
    from server.ml.audio_operations import execute_plan

    sr = 22050
    t = np.linspace(0, 4, sr * 4, endpoint=False)
    src = os.path.join(tempfile.gettempdir(), "chord_input.wav")
    sf.write(src, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)

    plan = regex_plan_from_prompt("add guitar chords: C G Am F")
    result = execute_plan(src, plan)
    assert result.get("output_path")
    assert result["metadata"]["chords"] == ["C", "G", "Am", "F"]
    y, _ = sf.read(result["output_path"])
    assert float(np.abs(y).max()) > 0.01
