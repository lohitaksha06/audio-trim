"""Deterministic rule-based parser used as the baseline.

This is intentionally a *separate, simplified* implementation. It follows the
same label schema as the learned models, so the comparison is fair, but it does
not import anything from the Audelle product.
"""

from __future__ import annotations

import re

from .schema import DEFAULT_FULL_MIX_OPS, Label

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

TARGET_PATTERNS: list[tuple[str, str]] = [
    ("speaker_2", r"\b(speaker\s*(2|two)|second\s+(speaker|voice)|guest)\b"),
    ("speaker_1", r"\b(speaker\s*(1|one)|first\s+(speaker|voice)|host|interviewer)\b"),
    ("vocals", r"\b(vocals?|voice|singer|singing|acapella|a cappella|background vocals|vocal track|dialogue|speech)\b"),
    ("drums", r"\b(drums?|kick|snare|hi-?hats?|cymbals?|percussion|drum fill)\b"),
    ("bass", r"\b(bass|bassline|bass line|sub|808)\b"),
    ("guitar", r"\b(guitars?|acoustic|electric guitar)\b"),
    ("keys", r"\b(piano|keys|keyboard|organ|synths?|synthesizer|rhodes)\b"),
    ("other", r"\b(fx|effects?|pads?|strings|brass|riser)\b"),
]

FULL_MIX_RE = re.compile(
    r"\b(everything|whole (mix|track|song)|entire (mix|track|song)|full mix|"
    r"the mix|the track|the song|the audio|overall|all speakers|everything else)\b"
)

NEGATIVE_AMOUNT_WORDS = re.compile(
    r"\b(down|lower|reduce|decrease|quieter|quiet down|drop|cut|minus|negative)\b"
)

REMOVE_VERBS = r"(remove|delete|take out|strip|get rid of|mute|drop|kill|cut out|pull out)"
KEEP_VERBS = r"(keep only|keep just|leave only|isolate|extract|give me|pull out)"
SOLO_VERB = r"\bsolo\s+(?:the\s+)?(?=(?:vocals?|drums?|bass|guitar|piano|keys|organ|synth))"
JUST_STEM = (
    r"\b(?:just|only)\s+(?:the\s+)?"
    r"(?:vocals?|voice|drums?|bass|guitar|piano|keys|organ|synth)\b"
)

WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "half": 0.5,
}


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------


def parse_timestamp(text: str) -> float | None:
    text = text.strip().lower()
    match = re.fullmatch(r"(\d+):(\d{2})(?:\.(\d+))?", text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        fraction = match.group(3) or ""
        return minutes * 60 + seconds + (int(fraction) / 10 ** len(fraction) if fraction else 0.0)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?", text)
    if match:
        return float(match.group(1))
    return None


TIME = r"(\d+:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds)?)"
RANGE_PATTERNS = [
    rf"\bfrom\s+{TIME}\s+to\s+{TIME}",
    rf"\bbetween\s+{TIME}\s+and\s+{TIME}",
    rf"\b{TIME}\s*(?:-|to)\s*{TIME}",
]
SINGLE_TIME_PATTERNS = [
    rf"\bfrom\s+{TIME}",
    rf"\bat\s+{TIME}",
]


def extract_span(prompt: str) -> list[float | None] | None:
    text = prompt.lower()
    for pattern in RANGE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            start = parse_timestamp(match.group(1))
            end = parse_timestamp(match.group(2))
            if start is not None and end is not None:
                return [start, end]
    match = re.search(r"\b(?:first|starting at)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?", text)
    if match:
        start = float(match.group(1))
        return [0.0, start] if "first" in match.group(0) else [start, None]
    match = re.search(r"\bstarting at\s+(\d+:\d{2}(?:\.\d+)?)", text)
    if match:
        start = parse_timestamp(match.group(1))
        length = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)\s+(?:long|clip)", text)
        if start is not None and length:
            return [start, start + float(length.group(1))]
        return [start, None] if start is not None else None
    for pattern in SINGLE_TIME_PATTERNS:
        match = re.search(pattern, text)
        if match:
            start = parse_timestamp(match.group(1))
            if start is not None:
                return [start, None]
    return None


# ---------------------------------------------------------------------------
# Target parsing
# ---------------------------------------------------------------------------


def extract_targets(prompt: str) -> list[str]:
    text = prompt.lower()
    found: list[str] = []
    for target, pattern in TARGET_PATTERNS:
        if re.search(pattern, text) and target not in found:
            found.append(target)
    if FULL_MIX_RE.search(text) and not found:
        found.append("full_mix")
    return found


def _canonicalise_speaker_targets(prompt: str) -> list[str]:
    """For remove_speaker, return the speaker(s) that should disappear."""
    text = prompt.lower()
    keep = re.search(KEEP_VERBS, text) or re.search(r"\b(only|just)\b", text)

    if re.search(r"\b(speaker\s*(2|two)|second\s+(speaker|voice)|guest)\b", text):
        mentioned = "speaker_2"
    elif re.search(r"\b(speaker\s*(1|one)|first\s+(speaker|voice)|host|interviewer)\b", text):
        mentioned = "speaker_1"
    else:
        return ["full_mix"]

    if keep and not re.search(r"\b(drop|remove|delete)\b", text):
        return ["speaker_1" if mentioned == "speaker_2" else "speaker_2"]
    return [mentioned]


# ---------------------------------------------------------------------------
# Amount parsing
# ---------------------------------------------------------------------------


def _number(token: str) -> float | None:
    token = token.strip().lower()
    if token in WORD_NUMBERS:
        return float(WORD_NUMBERS[token])
    try:
        return float(token)
    except ValueError:
        return None


def extract_amount(prompt: str, ops: list[str]) -> tuple[float | None, str | None]:
    text = prompt.lower()

    if "gain" in ops:
        match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*db", text)
        if match:
            value = float(match.group(1))
            if not match.group(1).startswith(("+", "-")) and NEGATIVE_AMOUNT_WORDS.search(text):
                value = -abs(value)
            return value, "dB"

    if "fade" in ops:
        match = re.search(r"(\d+(?:\.\d+)?|[a-z]+)\s*(?:s|sec|secs|second|seconds)\b", text)
        if match:
            value = _number(match.group(1))
            if value is not None:
                return value, "seconds"

    if "speed" in ops:
        match = re.search(r"(\d+(?:\.\d+)?)\s*bpm", text)
        if match:
            return float(match.group(1)), "bpm"
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\b", text)
        if match:
            return float(match.group(1)), "factor"
        match = re.search(r"(\d+(?:\.\d+)?)\s*(percent|%)\s*(faster|slower)", text)
        if match:
            amount = float(match.group(1))
            factor = 1 + amount / 100 if match.group(3) == "faster" else 1 - amount / 100
            return round(factor, 3), "factor"
        if re.search(r"\btwice\b|\bdouble\b", text):
            return 2.0, "factor"
        if re.search(r"\bhalf\b|\bhalve\b", text):
            return 0.5, "factor"

    if "remove_silence" in ops:
        match = re.search(r"(?:longer than|over|above|>)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?", text)
        if match:
            return float(match.group(1)), "seconds"

    return None, None


# ---------------------------------------------------------------------------
# Clause-level operation detection
# ---------------------------------------------------------------------------


def split_clauses(prompt: str) -> list[str]:
    parts = re.split(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\balso\b|\bplus\b|&)\s*", prompt.lower())
    return [p.strip() for p in parts if p.strip()]


def _is_speaker_op(text: str) -> bool:
    if not re.search(r"\b(speaker|host|interviewer|guest|voice)\b", text):
        return False
    return bool(
        re.search(rf"\b{REMOVE_VERBS}\b", text)
        or re.search(rf"\b{KEEP_VERBS}\b", text)
        or re.search(r"\b(only|just|drop|mute)\b", text)
    )


def detect_ops(clause: str) -> list[str]:
    text = clause.lower()
    ops: list[str] = []

    # 1. Speaker selection needs the verb + speaker noun.
    if _is_speaker_op(text):
        return ["remove_speaker"]

    # 2. Silence / pauses.
    if re.search(r"\b(silence|pauses?|dead air|gaps?|quiet parts?)\b", text):
        return ["remove_silence"]

    # 3. Denoise.
    if re.search(
        r"\b(noise|noisy|hiss|hum|static|buzz|crackle|denoise|room tone|wind|"
        r"background (sound|noise)|clean up|clean the audio)\b",
        text,
    ):
        return ["denoise"]

    # 4. Normalize / loudness matching.
    if re.search(
        r"\b(normaliz\w*|even out|level out|make the loudness|same (volume|level|loudness)|"
        r"match the loudness|balance the volume|bring everything to the same)\b",
        text,
    ):
        return ["normalize"]

    # 5. Speed / tempo.
    if re.search(r"\b(speed|tempo|bpm|faster|slower|playback|twice as fast|halve|1\.\d+x|\d+x)\b", text):
        return ["speed"]

    # 6. Fade.
    if re.search(r"\bfade\b", text):
        return ["fade"]

    # 7. Isolate (keep only a stem).
    if (
        re.search(rf"\b{KEEP_VERBS}\b", text)
        or re.search(SOLO_VERB, text)
        or re.search(JUST_STEM, text)
        or re.search(r"\bonly\b.*\b(remain|left)\b", text)
    ):
        if re.search(r"\b(vocals?|drums?|bass|guitar|piano|keys|organ|synth)\b", text):
            return ["isolate_stem"]

    # 8. Region words with cut/remove => cut_segment, not stem removal.
    if re.search(
        r"\b(solo|section|part|bridge|chorus|verse|intro|outro|fill|first|last)\b", text
    ) and re.search(r"\b(cut|remove|delete|take out|get rid of|trim|keep)\b", text):
        if re.search(r"\b(trim|crop|shorten|keep|cut it down to|only)\b", text):
            return ["trim_segment"]
        return ["cut_segment"]

    # 9. Explicit time range.
    if extract_span(text) is not None and re.search(r"\b(cut|remove|delete|trim|crop|keep|shorten)\b", text):
        if re.search(r"\b(trim|crop|shorten|keep)\b", text):
            return ["trim_segment"]
        return ["cut_segment"]

    # 10. Stem removal.
    if re.search(rf"\b{REMOVE_VERBS}\b", text) and extract_targets(text):
        return ["remove_stem"]

    # 11. Effects / mood.
    if re.search(
        r"\b(reverb|delay|echo|distortion|overdrive|chorus|filter|effect|saturation|"
        r"cinematic|darker|brighter|bigger|punch|cathedral|telephone|warm|lo-?fi)\b",
        text,
    ):
        return ["add_effect"]

    # 12. Level changes.
    if re.search(
        r"\b(db|louder|quieter|volume|gain|boost|turn up|turn down|raise|lower|"
        r"reduce|increase|level)\b",
        text,
    ):
        return ["gain"]

    return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rule_label(prompt: str) -> Label:
    """Parse one prompt into the same structured label shape as the dataset."""
    prompt = " ".join(prompt.split())
    clauses = split_clauses(prompt)
    ops: list[str] = []
    clause_targets: list[tuple[list[str], bool]] = []

    for clause in clauses:
        clause_ops = detect_ops(clause)
        targets = extract_targets(clause)
        if targets:
            clause_targets.append((targets, True))
        if not clause_ops:
            continue
        for op in clause_ops:
            if op not in ops:
                ops.append(op)

    if not ops:
        # No clause carried an operation; try the full prompt once.
        clause_ops = detect_ops(prompt)
        ops = clause_ops
        targets = extract_targets(prompt)
        clause_targets = [(targets, bool(targets))]

    # Targets: speaker operations are canonicalised, otherwise union the
    # explicit targets. If nothing is explicit, inherit concrete targets from
    # other clauses; only fall back to full_mix when there is nothing to inherit.
    if "remove_speaker" in ops:
        targets = _canonicalise_speaker_targets(prompt)
    else:
        concrete: list[str] = []
        any_explicit = False
        for targets, explicit in clause_targets:
            if explicit:
                any_explicit = True
                for target in targets:
                    if target not in concrete:
                        concrete.append(target)
        if concrete and any_explicit:
            targets = concrete
        else:
            targets = ["full_mix"]

    # If one operation defaults to full_mix but others name a stem, give the
    # unnamed operation the named stem instead ("keep only the vocals and add
    # reverb" -> both apply to vocals).
    named = [t for t in targets if t != "full_mix"]
    if named and "full_mix" in targets:
        only_explicit_full_mix = any(
            t == "full_mix" and explicit for t, explicit in clause_targets
        )
        if not only_explicit_full_mix:
            targets = [t for t in targets if t != "full_mix"]

    amount, unit = extract_amount(prompt, ops)
    if amount is not None and len(ops) > 1:
        amount, unit = None, None

    return Label(
        id="rule",
        text=prompt,
        ops=ops,
        targets=targets,
        amount=amount,
        unit=unit,
        span=extract_span(prompt),
        source="rules",
    )


def parse(prompt: str) -> dict:
    label = rule_label(prompt)
    return {
        "ops": label.ops,
        "targets": label.targets,
        "amount": label.amount,
        "unit": label.unit,
        "span": label.span,
    }
