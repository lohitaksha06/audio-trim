import pytest

from instruction_ml.rules import parse, parse_timestamp


@pytest.mark.parametrize(
    "prompt,ops,targets",
    [
        ("remove the vocals", ["remove_stem"], ["vocals"]),
        ("keep only the vocals", ["isolate_stem"], ["vocals"]),
        ("boost the bass by 4 db", ["gain"], ["bass"]),
        ("fade out the whole track over 4 seconds", ["fade"], ["full_mix"]),
        ("remove the section from 0:02 to 0:04", ["cut_segment"], ["full_mix"]),
        ("keep only from 0:05 to 0:20", ["trim_segment"], ["full_mix"]),
        ("remove speaker 2", ["remove_speaker"], ["speaker_2"]),
        ("only keep the first speaker", ["remove_speaker"], ["speaker_2"]),
        ("clean up the hiss", ["denoise"], ["full_mix"]),
        ("normalize the volume", ["normalize"], ["full_mix"]),
        ("speed it up", ["speed"], ["full_mix"]),
        ("add reverb to the vocals", ["add_effect"], ["vocals"]),
        ("remove the silence", ["remove_silence"], ["full_mix"]),
    ],
)
def test_rule_operations(prompt, ops, targets):
    result = parse(prompt)
    assert result["ops"] == ops
    assert set(result["targets"]) == set(targets)


def test_rule_multi_operation():
    result = parse("make the drums louder and remove the vocals")
    assert set(result["ops"]) == {"gain", "remove_stem"}
    assert set(result["targets"]) == {"drums", "vocals"}


def test_rule_amount_sign():
    assert parse("turn the vocals down by 3 db")["amount"] == -3.0
    assert parse("cut the bass by 6 db")["amount"] == -6.0
    assert parse("boost the bass by 4 db")["amount"] == 4.0


def test_rule_span():
    result = parse("cut from 0:05 to 0:20")
    assert result["span"] == [5.0, 20.0]
    assert parse_timestamp("2:30.5") == 150.5
    assert parse_timestamp("10") == 10.0


def test_rule_speed_units():
    assert parse("set tempo to 128 bpm")["unit"] == "bpm"
    assert parse("make it twice as fast")["amount"] == 2.0
    assert parse("make the track 10 percent faster")["amount"] == 1.1


def test_rule_unknown_returns_empty():
    assert parse("what is the weather tomorrow")["ops"] == []
