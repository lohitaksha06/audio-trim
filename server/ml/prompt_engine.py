import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    TRIM = "trim"
    REMOVE = "remove"
    SEPARATE = "separate"
    CONVERT = "convert"
    ISOLATE = "isolate"
    FADE = "fade"
    NORMALIZE = "normalize"
    REMOVE_SILENCE = "remove_silence"
    REMOVE_FILLERS = "remove_fillers"
    MOOD = "mood"
    SPEED = "speed"
    REVERB = "reverb"
    UNKNOWN = "unknown"


@dataclass
class PromptPlan:
    intent: Intent
    params: dict = field(default_factory=dict)
    raw_prompt: str = ""


def parse_timestamp(text: str) -> float | None:
    match = re.match(r"(\d+):(\d{2})(?:\.(\d+))?", text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        ms = int(match.group(3) or "0")
        return minutes * 60 + seconds + ms / 10 ** len(match.group(3) or "0")
    match = re.match(r"(\d+(?:\.\d+)?)s?", text)
    if match:
        return float(match.group(1))
    return None


def extract_time_range(prompt: str) -> tuple[float | None, float | None]:
    patterns = [
        r"from\s+(\d+:\d{2}(?:\.\d+)?)\s+to\s+(\d+:\d{2}(?:\.\d+)?)",
        r"between\s+(\d+:\d{2}(?:\.\d+)?)\s+and\s+(\d+:\d{2}(?:\.\d+)?)",
        r"(\d+:\d{2}(?:\.\d+)?)\s*[-–]\s*(\d+:\d{2}(?:\.\d+)?)",
        r"from\s+(\d+:\d{2}(?:\.\d+)?)",
        r"at\s+(\d+:\d{2}(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            groups = match.groups()
            start = parse_timestamp(groups[0])
            end = parse_timestamp(groups[1]) if len(groups) > 1 and groups[1] else None
            return start, end
    return None, None


def extract_instrument(prompt: str) -> str | None:
    instruments = {
        "vocals": ["vocal", "voice", "singer", "singing", "speech"],
        "drums": ["drum", "kick", "snare", "hi-hat", "hihat", "cymbal", "percussion"],
        "bass": ["bass", "bassline", "sub-bass"],
        "guitar": ["guitar", "acoustic guitar", "electric guitar", "strum"],
        "keys": ["piano", "keys", "keyboard", "synth", "synthesizer", "organ"],
        "other": ["pad", "strings", "orchestra", "fx", "effects"],
    }
    lower = prompt.lower()
    for stem, keywords in instruments.items():
        for kw in keywords:
            if kw in lower:
                return stem
    return None


_FILLER_PATTERN = re.compile(r"(?<![a-z])(um+|uh+|ah+|er+)s?(?![a-z])")


def classify_intent(prompt: str) -> Intent:
    lower = prompt.lower()

    if any(w in lower for w in ["trim", "cut", "crop", " shorten"]):
        return Intent.TRIM
    if any(w in lower for w in ["remove", "delete", "drop", "get rid of", "cut out"]):
        has_filler = "filler" in lower or _FILLER_PATTERN.search(lower)
        has_silence = any(w in lower for w in ["silence", "silent", "pause", "gap", "quiet part"])
        if has_filler or has_silence:
            return Intent.REMOVE_FILLERS if has_filler else Intent.REMOVE_SILENCE
        return Intent.REMOVE
    if any(w in lower for w in ["separate", "split", "stems", "stem"]):
        return Intent.SEPARATE
    if any(w in lower for w in ["convert", "export", "change format", "to mp3", "to wav", "to flac", "to aac", "to ogg"]):
        return Intent.CONVERT
    if any(w in lower for w in ["isolate", "keep only", "only the", "just the", "extract"]):
        return Intent.ISOLATE
    if any(w in lower for w in ["fade"]):
        return Intent.FADE
    if any(w in lower for w in ["normalize", "volume", "loudness", "level"]):
        return Intent.NORMALIZE
    if any(w in lower for w in ["darker", "brighter", "energetic", "calm", "mood", "feel", "tone"]):
        return Intent.MOOD
    if any(w in lower for w in ["speed", "tempo", "ramp"]) or re.search(
        r"\b(fast|faster|slow|slower|speed ?up|speed ?down)\b", lower
    ):
        return Intent.SPEED
    if any(w in lower for w in ["reverb", "echo", "delay", "space"]):
        return Intent.REVERB
    return Intent.UNKNOWN


def extract_format(prompt: str) -> str | None:
    formats = ["mp3", "wav", "flac", "aac", "ogg", "m4a"]
    lower = prompt.lower()
    for fmt in formats:
        if fmt in lower:
            return fmt
    return None


def extract_duration_seconds(prompt: str) -> float | None:
    match = re.search(r"(\d+)\s*sec(?:ond)?s?", prompt, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+)\s*min(?:ute)?s?", prompt, re.IGNORECASE)
    if match:
        return float(match.group(1)) * 60
    return None


def plan_from_prompt(prompt: str) -> PromptPlan:
    intent = classify_intent(prompt)
    start, end = extract_time_range(prompt)
    instrument = extract_instrument(prompt)
    fmt = extract_format(prompt)
    duration = extract_duration_seconds(prompt)

    params: dict = {}

    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end
    if instrument:
        params["instrument"] = instrument
    if fmt:
        params["format"] = fmt
    if duration:
        params["duration"] = duration

    if intent == Intent.MOOD:
        lower = prompt.lower()
        if "dark" in lower:
            params["mood"] = "dark"
        elif "bright" in lower:
            params["mood"] = "bright"
        elif "energetic" in lower or "energy" in lower:
            params["mood"] = "energetic"
        elif "calm" in lower or "chill" in lower:
            params["mood"] = "calm"

    if intent == Intent.FADE:
        lower = prompt.lower()
        params["fade_in"] = "in" in lower
        params["fade_out"] = "out" in lower

    if intent == Intent.SPEED:
        lower = prompt.lower()
        factor = 1.0
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\s*(?:as\s*)?(fast|faster|slow|slower)", lower)
        if m:
            num = float(m.group(1))
            factor = num if m.group(2).startswith("fast") else 1 / num
        elif re.search(r"\b(twice|double|2x|2x fast)\b", lower):
            factor = 2.0
        elif re.search(r"\b(half|slow|slower|slow ?down|speed ?down)\b", lower):
            factor = 0.75
        elif re.search(r"\b(fast|faster|speed ?up)\b", lower):
            factor = 1.5
        elif re.search(r"\bspeed\b", lower):
            factor = 1.5
        if factor != 1.0:
            params["speed_factor"] = factor

    if intent == Intent.REVERB:
        params["reverb_amount"] = 0.5

    return PromptPlan(intent=intent, params=params, raw_prompt=prompt)
