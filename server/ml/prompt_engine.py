import os
import re
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    TRIM = "trim"
    REMOVE = "remove"
    PAINT = "paint"
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
    ADD_INSTRUMENT = "add_instrument"
    COMBINE = "combine"
    BOOST = "boost"
    ENHANCE_VOCALS = "enhance_vocals"
    STYLE = "style"
    MIX_STEM = "mix_stem"
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


_INSTRUMENT_KEYWORDS = {
    "vocals": ["vocal", "voice", "singer", "singing", "speech"],
    "drums": ["drum", "kick", "snare", "hi-hat", "hihat", "cymbal", "percussion"],
    "bass": ["bass", "bassline", "sub-bass"],
    "guitar": ["guitar", "acoustic guitar", "electric guitar", "strum"],
    "keys": ["piano", "keys", "keyboard", "organ"],
    "strings": ["strings", "violin", "cello", "orchestra"],
    "synth": ["synth", "synthesizer", "synth pad", "synth lead", "supersaw", "saw", "pluck synth"],
    "tropical": ["tropical synth", "tropical pluck", "tropical edm synth", "marimba synth", "tropical"],
    "future": ["future synth", "futuristic synth", "future bass synth", "futuristic edm synth", "futuristic", "future bass"],
    "dubstep": ["dubstep", "wobble", "wobble bass", "dubstep bass", "riddim"],
    "edm": ["edm synth", "edm lead", "big room", "bigroom", "festival lead", "edm stab", "edm"],
    "other": ["pad", "fx", "effects"],
}


def extract_instrument(prompt: str) -> str | None:
    lower = prompt.lower()
    # "base guitar" = user's spelling of bass; word-boundary so "based" is safe
    if re.search(r"\bbase\b", lower):
        return "bass"
    # qualified synth kinds first — "tropical synth" is tropical, not generic synth
    for stem in ("tropical", "future", "dubstep", "edm"):
        for kw in _INSTRUMENT_KEYWORDS.get(stem, []):
            if kw in lower:
                return stem
    if "tropical" in lower:
        return "tropical"
    if "futuristic" in lower or "future bass" in lower or "future synth" in lower:
        return "future"
    if "dubstep" in lower or "wobble" in lower or "riddim" in lower:
        return "dubstep"
    if re.search(r"\bedm\b", lower) or "big room" in lower or "bigroom" in lower:
        # "edm drums" alone still means drums, but bare "add edm" means the edm kit
        if "drum" not in lower or "synth" in lower:
            return "edm"
    for stem, keywords in _INSTRUMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return stem
    return None


def extract_instruments(prompt: str) -> list[str]:
    """All instruments mentioned, in canonical order, de-duplicated."""
    lower = prompt.lower()
    found: list[str] = []
    if re.search(r"\bbase\b", lower):
        found.append("bass")
    # compound names are one instrument — don't double-count their parts
    guitar_scan = re.sub(r"\b(?:bass|base) guitar\b", " ", lower)
    other_scan = re.sub(r"\bsynth pad\b", "synth", lower)
    for stem, keywords in _INSTRUMENT_KEYWORDS.items():
        if stem in found:
            continue
        if stem == "guitar":
            scan = guitar_scan
        elif stem == "other":
            scan = other_scan
        else:
            scan = lower
        for kw in keywords:
            if kw in scan:
                found.append(stem)
                break
    # "tropical synth" is one instrument (tropical), not tropical+synth
    if "synth" in found and any(k in found for k in ("tropical", "future", "dubstep", "edm")):
        synth_tokens = len(re.findall(r"synth", lower))
        qual_tokens = len(
            re.findall(r"(?:tropical|future|futuristic|dubstep|edm|big(?:-| )?room)(?:\s+(?:edm|bass))?\s+synth", lower)
        )
        if synth_tokens <= qual_tokens:
            found.remove("synth")
    return found


_FILLER_PATTERN = re.compile(r"(?<![a-z])(um+|uh+|ah+|er+)s?(?![a-z])")


STYLE_KEYWORDS = {
    "house": ["house music", "house style", "four-on-the-floor", "four on the floor", "deep house"],
    "tropical": ["tropical", "tropical edm", "tropical house", "summer edm"],
    "edm": ["edm style", "edm drop", "festival edm", "big room", "bigroom", "big-room"],
    "futuristic": ["futuristic", "futuristic edm", "future bass", "future-bass", "future edm"],
    "dubstep": ["dubstep", "dubstep style", "wobble", "riddim", "brostep"],
    "lofi": ["lo-fi", "lofi", "chillhop"],
    "acoustic": ["acoustic style", "unplugged"],
}

ENHANCE_KEYWORDS = [
    "make voices clearer", "voices clearer", "enhance vocal", "clear vocal",
    "remove background noise", "remove disturbances", "denoise", "de-noise",
    "clean up voice", "vocal clarity", "reduce hiss", "remove hiss",
]


def extract_groove(prompt: str) -> str:
    lower = prompt.lower()
    if any(k in lower for k in ["dubstep", "wobble", "riddim", "brostep"]):
        return "dubstep"
    if any(k in lower for k in ["futuristic", "future bass", "future-bass", "future edm"]):
        return "future"
    if any(k in lower for k in ["tropical edm", "tropical house", "tropical"]):
        return "tropical"
    if any(k in lower for k in ["big room", "bigroom", "big-room", "festival edm"]):
        return "big_room"
    if re.search(r"\bedm\b", lower):
        return "big_room"
    if any(k in lower for k in ["swing", "shuffled", "shuffle"]):
        return "swing"
    if any(k in lower for k in ["four-on-the-floor", "four on the floor", "house groove", "four to the floor"]):
        return "four_on_floor"
    if any(k in lower for k in ["syncopat", "funky", "groovy", "offbeat groove"]):
        return "funky"
    if any(k in lower for k in ["half-time", "half time", "halftime", "laid back", "laid-back"]):
        return "half_time"
    if any(k in lower for k in ["double-time", "double time", "doubletime", "driving", "energetic groove"]):
        return "double_time"
    return "default"


def extract_style(prompt: str) -> str | None:
    lower = prompt.lower()
    for style, keys in STYLE_KEYWORDS.items():
        if any(k in lower for k in keys):
            return style
    # generic "convert ... into X style" capture
    m = re.search(r"(?:into|to|as)\s+(?:a\s+)?([a-z\- ]+?)\s*style", lower)
    if m:
        return m.group(1).strip().replace(" ", "_")[:32]
    return None


def classify_intent(prompt: str) -> Intent:
    lower = prompt.lower()

    # import-your-own-stem: "mix my uploaded stem", "import a stem", "blend this stem in"
    # (must run before SEPARATE so "stems" doesn't hijack it)
    if "stem" in lower and any(
        w in lower for w in ["import", "my stem", "my own", "own stem", "uploaded", "mix in", "blend", "mix my", "mix this", "add my stem"]
    ):
        return Intent.MIX_STEM

    if any(k in lower for k in ENHANCE_KEYWORDS):
        return Intent.ENHANCE_VOCALS
    style = extract_style(prompt)
    if style and any(w in lower for w in ["convert", "make it", "turn into", "style", "remix"]):
        return Intent.STYLE

    if any(
        w in lower
        for w in ["cant hear", "can't hear", "cannot hear", "too quiet", "too soft", "louder", "turn it up", "turn up the", "bring up the", "pump up", "boost the", "boost "]
    ):
        return Intent.BOOST

    # combine first: "add both drums and bass" must beat the single-add branch
    mentioned = extract_instruments(prompt)
    if "combin" in lower or "comebin" in lower:
        if mentioned:
            return Intent.COMBINE
    if (
        any(w in lower for w in ["both", "together", "merge", "mix them", "layer them"])
        and len(mentioned) >= 2
    ):
        return Intent.COMBINE

    if any(w in lower for w in ["add ", "generate", "create", "insert", "layer", "synthesize"]):
        # "add bass / synth / drums" must be checked before remove/isolate
        if len(mentioned) >= 2:
            return Intent.COMBINE
        if extract_instrument(prompt):
            return Intent.ADD_INSTRUMENT
        if any(k in lower for k in ["bass", "drum", "synth", "guitar", "piano", "keys", "pad", "strings", "tropical", "future", "futuristic", "dubstep", "wobble", "edm", "big room", "bigroom", "lead", "pluck", "supersaw", "riddim"]):
            return Intent.ADD_INSTRUMENT
    if any(w in lower for w in ["trim", "cut", "crop", "shorten"]):
        return Intent.TRIM
    if any(w in lower for w in ["inpaint", "fill smoothly", "seamless", "paint over", "fill the gap", "clean up the", "remove the cough", "fix that"]):
        return Intent.PAINT
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


def regex_plan_from_prompt(prompt: str) -> PromptPlan:
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

    if intent == Intent.ADD_INSTRUMENT:
        # ensure we have an instrument even if keyword was "synth" mapped to keys
        if not params.get("instrument"):
            lower = prompt.lower()
            if "dubstep" in lower or "wobble" in lower or "riddim" in lower:
                params["instrument"] = "dubstep"
            elif "tropical" in lower:
                params["instrument"] = "tropical"
            elif "futur" in lower:
                params["instrument"] = "future"
            elif "big room" in lower or "bigroom" in lower or ("edm" in lower and "style" not in lower):
                params["instrument"] = "edm"
            elif "synth" in lower or "supersaw" in lower or "lead" in lower or "pluck" in lower:
                params["instrument"] = "synth"
            elif "pad" in lower:
                params["instrument"] = "keys"
            elif "bass" in lower:
                params["instrument"] = "bass"
            elif "drum" in lower:
                params["instrument"] = "drums"
            elif "guitar" in lower:
                params["instrument"] = "guitar"
            else:
                params["instrument"] = "other"
        params["groove"] = extract_groove(prompt)
        m_bpm = re.search(r"(\d{2,3})\s*bpm", prompt.lower())
        if m_bpm:
            params["target_bpm"] = float(m_bpm.group(1))

    if intent == Intent.STYLE:
        style = extract_style(prompt)
        if style:
            params["style"] = style
        params["groove"] = extract_groove(prompt)

    if intent == Intent.ENHANCE_VOCALS:
        lower = prompt.lower()
        params["denoise"] = any(k in lower for k in ["noise", "hiss", "disturbance", "denoise", "clean"])
        params["clarity"] = True

    if intent == Intent.COMBINE:
        params["instruments"] = extract_instruments(prompt) or ["drums", "bass"]
        # keep only generatable layers (drop vocals etc. that can't be synthesized)
        generatable = {"drums", "bass", "guitar", "keys", "strings",
                       "synth", "tropical", "future", "dubstep", "edm", "other"}
        params["instruments"] = [i for i in params["instruments"] if i in generatable] or ["drums", "bass"]
        params["groove"] = extract_groove(prompt)
        m_bpm = re.search(r"(\d{2,3})\s*bpm", prompt.lower())
        if m_bpm:
            params["target_bpm"] = float(m_bpm.group(1))

    if intent == Intent.MIX_STEM:
        # stem file is supplied via API (stem_path); prompt may carry mix level
        m = re.search(r"(\d{1,3})\s*%", prompt)
        if m:
            params["stem_level"] = min(100, max(5, int(m.group(1)))) / 100.0
        if "quiet" in prompt.lower() or "background" in prompt.lower():
            params.setdefault("stem_level", 0.35)
        params["groove"] = extract_groove(prompt)

    if intent == Intent.BOOST:
        params["target"] = extract_instrument(prompt) or "other"

    return PromptPlan(intent=intent, params=params, raw_prompt=prompt)


def plan_from_prompt(prompt: str) -> PromptPlan:
    """LLM-first prompt parsing with a deterministic regex fallback.

    When the environment enables it (``USE_LLM=1``), a configured LLM parses
    the prompt into a structured plan. Otherwise (or on failure) the regex
    engine is used so the pipeline keeps working offline.
    """
    if os.environ.get("USE_LLM") == "1":
        from server.ml.llm.prompt_llm import llm_plan_from_prompt

        plan = llm_plan_from_prompt(prompt)
        if plan is not None:
            return plan
    return regex_plan_from_prompt(prompt)
