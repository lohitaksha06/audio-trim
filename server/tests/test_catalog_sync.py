"""The UI catalogue must match what the backend can actually render.

`website/src/app/features/instrumentCatalog.ts` is the list of instruments and
beats the product advertises. If it drifts from the trained model or the drum
bank, the app starts promising sounds it cannot produce -- which is exactly the
class of dishonesty this project is trying to remove.

So this test parses the TS file and compares it against the real backend state.
"""

import re
from pathlib import Path

import pytest

from server.ml.synthesis import drums, timbre

CATALOG = (
    Path(__file__).resolve().parents[2]
    / "website"
    / "src"
    / "app"
    / "features"
    / "instrumentCatalog.ts"
)


def _read() -> str:
    if not CATALOG.exists():
        pytest.skip(f"catalogue not found at {CATALOG}")
    return CATALOG.read_text(encoding="utf-8")


def _block(name: str) -> str:
    """Extract the array literal assigned to `name`.

    Starts from the `=` so the `[]` in a type annotation (`Instrument[]`) is
    not mistaken for the array body.
    """
    text = _read()
    start = text.index(name)
    eq = text.index("=", start)
    start = text.index("[", eq)
    depth, i = 0, start
    while i < len(text):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    raise AssertionError(f"unterminated array for {name}")


def _fields(block: str, key: str) -> list[str]:
    """All `key: "value"` string values inside a catalogue block."""
    return re.findall(rf'{key}:\s*"([^"]+)"', block)


class TestCatalogMatchesBackend:
    def test_every_advertised_instrument_has_a_trained_family(self):
        trained = {f.split("/")[0] for f in timbre.families()}
        advertised = _fields(_block("TRAINED_INSTRUMENTS"), "family")
        assert advertised, "no instruments advertised"
        missing = [f for f in advertised if f not in trained]
        assert not missing, (
            f"UI advertises families with no trained timbre: {missing}. "
            f"Trained: {sorted(trained)}"
        )

    def test_catalog_covers_every_trained_family(self):
        """Not just a subset check - nothing trained should be unreachable."""
        trained = {f.split("/")[0] for f in timbre.families()}
        advertised = set(_fields(_block("TRAINED_INSTRUMENTS"), "family"))
        missing = sorted(trained - advertised)
        assert not missing, (
            f"trained but not offered in the UI: {missing}. Add them to "
            "instrumentCatalog.ts or retrain without them."
        )

    def test_every_button_routes_to_add_instrument(self):
        """Every prompt the UI can send must reach the instrument pipeline.

        The catalogue holds full prompts (not fragments the UI wraps), so this
        exercises exactly what the user can press.
        """
        from server.ml.prompt_engine import Intent, regex_plan_from_prompt

        wrong = []
        for block_name in ("TRAINED_INSTRUMENTS", "DSP_INSTRUMENTS", "BEAT_STYLES"):
            for prompt in _fields(_block(block_name), "prompt"):
                plan = regex_plan_from_prompt(prompt)
                if plan.intent != Intent.ADD_INSTRUMENT:
                    wrong.append((prompt, plan.intent.value))
        assert not wrong, (
            f"UI buttons that do not reach the instrument pipeline: {wrong}"
        )

    def test_trained_buttons_land_on_a_trained_family(self):
        """Each trained button must actually reach the model, not a fallback."""
        from server.ml.prompt_engine import Intent, regex_plan_from_prompt

        for prompt in _fields(_block("TRAINED_INSTRUMENTS"), "prompt"):
            plan = regex_plan_from_prompt(prompt)
            assert plan.intent == Intent.ADD_INSTRUMENT, prompt
            instrument = plan.params.get("instrument")
            assert instrument, f"{prompt} produced no instrument"
            assert timbre._nearest_family(instrument) is not None, (
                f"{prompt} resolves to instrument '{instrument}', which has no "
                "trained family — it would silently fall back to DSP"
            )

    def test_dsp_instruments_are_honestly_labelled(self):
        """Anything not from the trained model must say so in the UI copy."""
        block = _block("DSP_INSTRUMENTS")
        blurbs = _fields(block, "blurb")
        assert blurbs, "no DSP instruments listed"
        for b in blurbs:
            assert "not trained" in b.lower(), (
                f"DSP instrument blurb does not disclose it is untrained: '{b}'"
            )

    def test_every_advertised_beat_style_exists(self):
        styles = set(drums.styles())
        advertised = _fields(_block("BEAT_STYLES"), "prompt")
        assert advertised, "no beats advertised"
        unknown = [p for p in advertised if drums.resolve_style(p) not in styles]
        assert not unknown, f"UI advertises beat styles that do not exist: {unknown}"

    def test_advertised_drum_types_are_in_the_bank(self):
        labels = set(drums.kit())
        entries = re.findall(r'"([a-z_]+)"', _block("DRUM_TYPES"))
        assert entries, "no drum types advertised"
        missing = [d for d in entries if d not in labels]
        assert not missing, f"UI lists drum types absent from the bank: {missing}"

    def test_every_bank_drum_type_is_reachable_in_the_ui(self):
        advertised = set(re.findall(r'"([a-z_]+)"', _block("DRUM_TYPES")))
        missing = sorted(set(drums.kit()) - advertised)
        assert not missing, f"bank contains drum types the UI never mentions: {missing}"

    def test_no_instrument_is_promised_without_a_blurb(self):
        for block_name in ("TRAINED_INSTRUMENTS", "DSP_INSTRUMENTS", "BEAT_STYLES"):
            block = _block(block_name)
            labels = _fields(block, "label")
            blurbs = _fields(block, "blurb")
            assert len(labels) == len(blurbs), (
                f"{block_name}: {len(labels)} labels but {len(blurbs)} blurbs"
            )