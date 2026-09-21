"""Prompt understanding benchmark — paraphrase coverage for NLU eval.

100% deterministic (regex engine). Measures intent accuracy + param extraction
so the team and paper reviewers can track NLU quality without an LLM key.

Design: every entry has `prompt`, `intent` (Intent.value), and optional
`params` subset to verify. `evaluate_prompt_bench()` runs the current
`regex_plan_from_prompt` over BENCH and returns accuracy + per-intent breakdown.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from server.ml.prompt_engine import Intent, regex_plan_from_prompt

# 36 entries — covers all intents with paraphrases, edge cases, filler-vs-drum bug
BENCH: list[dict[str, Any]] = [
    # TRIM (4)
    {"prompt": "trim the intro", "intent": "trim"},
    {"prompt": "cut from 0:05 to 0:20", "intent": "trim", "params": {"start": 5.0, "end": 20.0}},
    {"prompt": "crop from 1:30 to 2:00", "intent": "trim"},
    {"prompt": "shorten from 0:02 to 0:08", "intent": "trim"},
    # REMOVE (4)
    {"prompt": "remove the section from 0:02 to 0:04", "intent": "remove"},
    {"prompt": "remove the drums", "intent": "remove", "params": {"instrument": "drums"}},
    {"prompt": "remove the kick drum from 2:30 to 3:45", "intent": "remove"},
    {"prompt": "delete the bass between 1:00 and 1:30", "intent": "remove"},
    # REMOVE_FILLERS / SILENCE
    {"prompt": "remove ums and ahs", "intent": "remove_fillers"},
    {"prompt": "remove filler words", "intent": "remove_fillers"},
    {"prompt": "remove the silence", "intent": "remove_silence"},
    {"prompt": "remove pauses longer than 0.5s", "intent": "remove_silence"},
    # SEPARATE (2)
    {"prompt": "separate into stems", "intent": "separate"},
    {"prompt": "split the audio into drums and vocals", "intent": "separate"},
    # CONVERT (2)
    {"prompt": "convert to mp3", "intent": "convert", "params": {"format": "mp3"}},
    {"prompt": "export as wav", "intent": "convert", "params": {"format": "wav"}},
    # ISOLATE (2)
    {"prompt": "keep only the vocals", "intent": "isolate", "params": {"instrument": "vocals"}},
    {"prompt": "extract just the bass", "intent": "isolate"},
    # FADE (2)
    {"prompt": "fade in and out", "intent": "fade"},
    {"prompt": "add a fade in at the start", "intent": "fade"},
    # NORMALIZE (2)
    {"prompt": "normalize the volume", "intent": "normalize"},
    {"prompt": "make the loudness even", "intent": "normalize"},
    # MOOD (3)
    {"prompt": "make it darker", "intent": "mood", "params": {"mood": "dark"}},
    {"prompt": "make it energetic and bright", "intent": "mood"},
    {"prompt": "calm this down a bit", "intent": "mood"},
    # SPEED (4)
    {"prompt": "speed it up", "intent": "speed", "params": {"speed_factor": 1.5}},
    {"prompt": "make it twice as fast", "intent": "speed", "params": {"speed_factor": 2.0}},
    {"prompt": "slow it down", "intent": "speed"},
    {"prompt": "2x faster please", "intent": "speed"},
    # REVERB (2)
    {"prompt": "add reverb", "intent": "reverb"},
    {"prompt": "make it echo like a cathedral", "intent": "reverb"},
    # PAINT (2)
    {"prompt": "remove that cymbal crash and fill smoothly at 1:23", "intent": "paint"},
    {"prompt": "clean up the cough at 0:03 and paint over it", "intent": "paint"},
    # ADD_INSTRUMENT (4)
    {"prompt": "add a bass line", "intent": "add_instrument", "params": {"instrument": "bass"}},
    {"prompt": "add synth pad in the bridge", "intent": "add_instrument"},
    {"prompt": "add funky syncopated drums following the groove", "intent": "add_instrument", "params": {"instrument": "drums"}},
    {"prompt": "add four-on-the-floor drums", "intent": "add_instrument", "params": {"groove": "four_on_floor"}},
    # STYLE (2)
    {"prompt": "convert this song into a house music style", "intent": "style", "params": {"style": "house"}},
    {"prompt": "make it tropical edm style", "intent": "style"},
    # ENHANCE_VOCALS (2)
    {"prompt": "make voices clearer and remove background noise", "intent": "enhance_vocals"},
    {"prompt": "enhance vocals and denoise", "intent": "enhance_vocals"},
    # ENHANCE_VOCALS voice-audibility + combined voice-first chaining
    {"prompt": "make her voice audible and clear", "intent": "enhance_vocals"},
    {"prompt": "add drums to this and make the voice more clear, remove the background noises", "intent": "enhance_vocals", "params": {"add_instrument": "drums"}},
    # selective drum pieces
    {"prompt": "add only snare", "intent": "add_instrument", "params": {"instrument": "drums", "drum_parts": ["snare"]}},
    {"prompt": "add a kick drum", "intent": "add_instrument", "params": {"instrument": "drums", "drum_parts": ["kick"]}},
    # new-school grooves
    {"prompt": "add uk garage drums", "intent": "add_instrument", "params": {"instrument": "drums", "groove": "garage"}},
    {"prompt": "add a 2-step beat", "intent": "add_instrument", "params": {"groove": "garage"}},
    {"prompt": "add amapiano drums", "intent": "add_instrument", "params": {"groove": "amapiano"}},
    {"prompt": "add an afro house groove", "intent": "add_instrument", "params": {"groove": "afro_house"}},
    {"prompt": "add a jungle break", "intent": "add_instrument", "params": {"groove": "jungle"}},
    {"prompt": "add a grime beat", "intent": "add_instrument", "params": {"groove": "grime"}},
    {"prompt": "convert to garage style", "intent": "style", "params": {"style": "garage"}},
    # new editing options
    {"prompt": "reverse the intro", "intent": "reverse"},
    {"prompt": "loop the chorus 3 times", "intent": "repeat", "params": {"times": 3}},
    {"prompt": "pitch it up 2 semitones", "intent": "transpose", "params": {"semitones": 2.0}},
    # COMBINE (3)
    {"prompt": "combine both drums and bass", "intent": "combine"},
    {"prompt": "can u comebin both drum and base", "intent": "combine"},
    {"prompt": "add a base guitar in the back too", "intent": "add_instrument", "params": {"instrument": "bass"}},
    # BOOST (2)
    {"prompt": "i cant hear the drums", "intent": "boost", "params": {"target": "drums"}},
    {"prompt": "make the drums louder", "intent": "boost"},
    # GROOVE/EXTRA (2)
    {"prompt": "add swing drums", "intent": "add_instrument", "params": {"groove": "swing"}},
    {"prompt": "add strings", "intent": "add_instrument"},
    # UNKNOWN (1)
    {"prompt": "hello world", "intent": "unknown"},
]


def evaluate_prompt_bench() -> dict[str, Any]:
    """Run regex engine over BENCH and return accuracy + per-intent breakdown."""
    total = len(BENCH)
    correct = 0
    per_intent: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    failures: list[dict[str, Any]] = []
    param_checks = 0
    param_correct = 0

    for item in BENCH:
        prompt = item["prompt"]
        expected_intent = item["intent"]
        plan = regex_plan_from_prompt(prompt)
        got = plan.intent.value
        per_intent[expected_intent]["total"] += 1
        if got == expected_intent:
            correct += 1
            per_intent[expected_intent]["correct"] += 1
            # param subset check if specified
            expected_params = item.get("params", {})
            for k, v in expected_params.items():
                param_checks += 1
                if plan.params.get(k) == v:
                    param_correct += 1
                else:
                    failures.append({"prompt": prompt, "param": k, "expected": v, "got": plan.params.get(k)})
        else:
            failures.append({"prompt": prompt, "expected": expected_intent, "got": got})

    accuracy = correct / total if total else 0.0
    param_accuracy = (param_correct / param_checks) if param_checks else None

    # convert defaultdict to plain dict
    per_intent_plain = {
        k: {"accuracy": (v["correct"] / v["total"] if v["total"] else 0), **v}
        for k, v in per_intent.items()
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "param_checks": param_checks,
        "param_accuracy": round(param_accuracy, 4) if param_accuracy is not None else None,
        "per_intent": per_intent_plain,
        "failures": failures,
    }
