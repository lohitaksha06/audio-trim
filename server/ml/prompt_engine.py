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
    GAIN = "gain"
    REMOVE_SILENCE = "remove_silence"
    REMOVE_FILLERS = "remove_fillers"
    MOOD = "mood"
    SPEED = "speed"
    REVERB = "reverb"
    ADD_INSTRUMENT = "add_instrument"
    COMBINE = "combine"
    BOOST = "boost"
    MIX = "mix"
    ENHANCE_VOCALS = "enhance_vocals"
    STYLE = "style"
    MIX_STEM = "mix_stem"
    REVERSE = "reverse"
    REPEAT = "repeat"
    TRANSPOSE = "transpose"
    UNKNOWN = "unknown"


# Canonical instrument catalog shown in the Guide / Mix Lab.
# key -> (display name, family, one-line how-it-sounds)
INSTRUMENT_CATALOG: dict[str, tuple[str, str, str]] = {
    "vocals": ("Vocals", "voice", "Lead vocal / speech presence"),
    "drums": ("Drums", "rhythm", "Kick + snare + hats, beat-synced"),
    "bass": ("Bass guitar", "bass", "Plucky electric bass line in key"),
    "808": ("808 bass", "bass", "Deep sliding sub for trap / hip-hop"),
    "guitar": ("Guitar", "strings", "Strummed triad in key, 8th-note groove"),
    "piano": ("Piano", "keys", "Warm keys triad with bar pulse"),
    "keys": ("Keys / Organ", "keys", "Warm organ-style pad with pulse"),
    "organ": ("Organ", "keys", "Sustained organ chords"),
    "strings": ("Strings", "orchestral", "Sustained section, slow bowed swell"),
    "violin": ("Violin", "orchestral", "Legato lead line with vibrato"),
    "cello": ("Cello", "orchestral", "Low legato counter-line"),
    "brass": ("Brass section", "orchestral", "Stabby section hits on beats"),
    "trumpet": ("Trumpet", "orchestral", "Bright lead stabs"),
    "sax": ("Saxophone", "orchestral", "Breathy lead with vibrato"),
    "flute": ("Flute", "orchestral", "Airy lead line, soft attack"),
    "harp": ("Harp", "orchestral", "Glissando arpeggios"),
    "marimba": ("Marimba", "percussion", "Sunny mallet plucks, offbeat"),
    "choir": ("Choir pad", "voice", "Soft vocal-style pad"),
    "synth": ("Warm synth", "synth", "Pad + gentle arp"),
    "supersaw": ("Supersaw lead", "synth", "Stacked-saw festival lead"),
    "square_lead": ("Square lead", "synth", "Chiptune / synthwave square lead"),
    "pluck": ("Pluck", "synth", "Short tropical-style pluck"),
    "pad": ("Pad", "synth", "Slow-attack warm pad"),
    "arp": ("Arp", "synth", "16th-note arpeggiator in key"),
    "acid": ("Acid bass", "synth", "Resonant squelchy 303-style bassline"),
    "reese": ("Reese bass", "synth", "Detuned DnB / dubstep reese"),
    "tropical": ("Tropical synth", "edm", "Marimba-like sunny pluck"),
    "future": ("Future bass", "edm", "Supersaw chords with sidechain pump"),
    "dubstep": ("Dubstep wobble", "edm", "LFO wobble bass, half-time"),
    "edm": ("Big-room EDM", "edm", "Festival stab on every beat + sub pulse"),
    "house": ("House", "edm", "Four-on-the-floor + offbeat hats"),
    "deep_house": ("Deep house", "edm", "Warm Rhodes-ish chords, low-slung groove"),
    "techno": ("Techno", "edm", "Driving 4/4 + rolling sub + dark stab"),
    "trance": ("Trance", "edm", "Uplifting gated supersaw + rolling bass"),
    "trap": ("Trap", "edm", "Half-time 808 + fast hats + dark bell"),
    "dnb": ("Drum & bass", "edm", "Fast breakbeat + reese sub"),
    "hardstyle": ("Hardstyle", "edm", "Punchy distorted kick + screech lead"),
    "phonk": ("Phonk", "edm", "Cowbell lead + Memphis 808 + hats"),
    "synthwave": ("Synthwave", "edm", "Retro square/saw pad + gated drums"),
    "garage": ("UK garage", "edm", "Shuffled 2-step drums + warpy sub"),
    "amapiano": ("Amapiano", "edm", "Shaker groove + log-drum bassline"),
    "afro_house": ("Afro house", "edm", "Syncopated percussion + deep groove"),
    "jungle": ("Jungle", "edm", "Chopped Amen break + heavy sub"),
    "grime": ("Grime", "edm", "Half-time eski beat + deep sub"),
    "lofi_keys": ("Lo-fi keys", "edm", "Dusty Rhodes + soft half-time drums"),
    "other": ("FX / Pad", "fx", "Generic texture layer"),
}

# Synth oscillator / timbre waves users can name ("add a supersaw", "square wave bass").
WAVE_KEYWORDS: dict[str, list[str]] = {
    "supersaw": ["supersaw", "super saw", "stacked saw", "festival lead"],
    "saw": ["saw wave", "sawtooth", "saw bass", "saw lead"],
    "square": ["square wave", "square lead", "square bass", "chiptune"],
    "triangle": ["triangle wave", "triangle lead", "triangle bass"],
    "sine": ["sine wave", "sine lead", "sine bass", "sub sine"],
    "fm": ["fm synth", "fm bass", "fm lead", "fm electric piano", "dx7"],
    "wavetable": ["wavetable", "morphing synth", "serum"],
    "acid": ["acid", "303", "tb-303", "squelch"],
    "reese": ["reese", "reese bass", "detuned bass"],
    "hoover": ["hoover", "hoover lead", "dominator"],
    "pluck": ["pluck", "plucked synth", "pluck bass"],
    "808": ["808", "eight-oh-eight", "sub bass", "sub-bass", "slide bass"],
    "pad": ["pad", "atmosphere pad", "ambient pad"],
    "arp": ["arp", "arpeggio", "arpeggiator", "sequenced synth"],
}

# Demucs 4-stem names — the mixer works on these; everything else folds into "other".
MIX_STEMS = ("vocals", "drums", "bass", "other")


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
    "vocals": ["vocal", "voice", "singer", "singing", "speech", "acapella", "choir lead"],
    "drums": ["drum", "kick", "snare", "hi-hat", "hihat", "cymbal", "percussion", "breakbeat", "break beat"],
    "bass": ["bass guitar", "bassline", "bass line", "electric bass", "bass"],
    "808": ["808", "eight-oh-eight", "sub bass", "sub-bass", "slide bass", "trap bass"],
    "guitar": ["guitar", "acoustic guitar", "electric guitar", "strum", "ukulele", "banjo"],
    "piano": ["piano", "rhodes", "electric piano", "grand piano"],
    "keys": ["keys", "keyboard"],
    "organ": ["organ", "hammond", "church organ"],
    "strings": ["strings", "orchestra", "string section"],
    "violin": ["violin", "fiddle", "viola"],
    "cello": ["cello"],
    "brass": ["brass", "brass section", "horns", "horn section"],
    "trumpet": ["trumpet"],
    "sax": ["sax", "saxophone"],
    "flute": ["flute", "piccolo"],
    "harp": ["harp", "glissando"],
    "marimba": ["marimba", "mallet", "kalimba", "xylophone"],
    "choir": ["choir", "choir pad", "vocal pad", "ahh pad", "ooh pad"],
    "synth": ["synth", "synthesizer", "synth lead", "synthwave lead"],
    "supersaw": ["supersaw", "super saw", "stacked saw", "festival lead"],
    "square_lead": ["square lead", "square wave", "chiptune", "8-bit", "8bit"],
    "pluck": ["pluck", "plucked synth", "pluck synth", "pluck bass", "tropical pluck"],
    "pad": ["pad", "atmosphere pad", "ambient pad", "warm pad", "synth pad"],
    "arp": ["arp", "arpeggio", "arpeggiator", "sequenced synth"],
    "acid": ["acid", "303", "tb-303", "squelch", "acid bass"],
    "reese": ["reese", "reese bass", "detuned bass"],
    "tropical": ["tropical synth", "tropical edm synth", "marimba synth", "tropical"],
    "future": ["future synth", "futuristic synth", "future bass synth", "futuristic edm synth", "futuristic", "future bass"],
    "dubstep": ["dubstep", "wobble", "wobble bass", "dubstep bass", "riddim", "brostep"],
    "edm": ["edm synth", "edm lead", "big room", "bigroom", "festival lead", "edm stab", "edm"],
    "house": ["house synth", "house bass", "house drums", "house music", "house", "tech house", "french house"],
    "deep_house": ["deep house"],
    "techno": ["techno", "peak-time techno", "melodic techno", "rolling techno"],
    "trance": ["trance", "uplifting trance", "psytrance", "psy-trance"],
    "trap": ["trap synth", "trap drums", "trap hats", "trap edm", "hybrid trap", "trap"],
    "dnb": ["drum and bass", "drum & bass", "dnb", "liquid dnb"],
    "hardstyle": ["hardstyle", "hard style", "rawstyle"],
    "phonk": ["phonk", "drift phonk", "house phonk", "cowbell"],
    "synthwave": ["synthwave", "retrowave", "outrun", "synth-wave", "retro synth"],
    "garage": ["uk garage", "garage", "2-step", "2step", "two-step", "two step", "speed garage", "garage drums", "garage beat"],
    "amapiano": ["amapiano", "amapiano", "log drum", "logdrum", "amapiano beat"],
    "afro_house": ["afro house", "afrohouse", "afro-house", "afro house beat"],
    "jungle": ["jungle", "jungle drums", "amen break"],
    "grime": ["grime", "grime beat", "eskibeat", "eski beat", "eski"],
    "lofi_keys": ["lofi", "lo-fi", "chillhop", "lofi keys", "dusty keys"],
    "other": ["fx", "effects", "riser", "downlifter", "impact", "sfx"],
}


_QUALIFIED_KINDS = (
    "tropical", "future", "dubstep", "edm", "deep_house", "afro_house", "house", "techno",
    "trance", "trap", "dnb", "hardstyle", "phonk", "synthwave",
    "garage", "amapiano", "jungle", "grime",
    "808", "acid", "reese", "choir",
    "supersaw", "square_lead", "pluck", "pad", "arp",
)


def extract_wave(prompt: str) -> str | None:
    """Named oscillator/timbre wave, e.g. 'supersaw', 'square', 'acid', '808'."""
    lower = prompt.lower()
    for wave, kws in WAVE_KEYWORDS.items():
        for kw in kws:
            if kw in lower:
                return wave
    return None


_GROOVE_KINDS = {"tropical", "future", "dubstep", "edm", "house", "deep_house",
                 "techno", "trance", "trap", "dnb", "hardstyle", "phonk", "synthwave",
                 "garage", "amapiano", "afro_house", "jungle", "grime"}


def extract_instrument(prompt: str) -> str | None:
    lower = prompt.lower()
    # "base guitar" = user's spelling of bass; word-boundary so "based" is safe
    if re.search(r"\bbase\b", lower):
        return "bass"
    # "techno drums" / "hardstyle kick" = drums played with that groove, not a synth kit
    if any(w in lower for w in ("drum", "kick", "snare", "hat", "cymbal", "percussion")):
        others = [s for s in extract_instruments(prompt) if s not in _GROOVE_KINDS and s != "drums"]
        if not others:
            return "drums"
    # qualified synth/EDM kinds first — "tropical synth" is tropical, not generic synth
    for stem in _QUALIFIED_KINDS:
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
    # "reese bass" / "acid bassline" / "808" are synth subs, not bass guitar
    bass_scan = re.sub(
        r"\b(?:reese|acid|fm|trap|sub|slide|pluck|saw|square|808|synth)\s*bass(?:line| line)?", " ", lower
    )
    # "synthwave" / "house" kits contain "synth" as a substring — not a warm synth
    synth_scan = re.sub(
        r"\b(?:tropical|future|futuristic|dubstep|edm|big(?:-| )?room|house|techno|trance|trap|dnb|jungle|hardstyle|phonk|synthwave|retrowave|outrun|garage|amapiano|afro(?:-| )?house|grime)\b", " ", lower
    )
    # "amapiano" contains the substring "piano" — not an acoustic piano
    piano_scan = re.sub(r"amapiano", " ", lower)
    for stem, keywords in _INSTRUMENT_KEYWORDS.items():
        if stem in found:
            continue
        if stem == "guitar":
            scan = guitar_scan
        elif stem == "other":
            scan = other_scan
        elif stem == "bass":
            scan = bass_scan
        elif stem == "synth":
            scan = synth_scan
        elif stem == "piano":
            scan = piano_scan
        else:
            scan = lower
        for kw in keywords:
            if kw in scan:
                found.append(stem)
                break
    # "choir pad" is one instrument (choir); "tropical pluck" is tropical;
    # "synth pad" is one warm pad, not synth + pad;
    # "synthwave pad" is the retro kit (its layer already is pad + arp)
    if "choir" in found and "pad" in found and "choir pad" in lower:
        found.remove("pad")
    if "tropical" in found and "pluck" in found and "tropical pluck" in lower:
        found.remove("pluck")
    if "synth" in found and "pad" in found and "synth pad" in lower:
        found.remove("synth")
    if "synthwave" in found and "pad" in found and "synthwave pad" in lower:
        found.remove("pad")
    # "tropical synth" is one instrument (tropical), not tropical+synth
    if "synth" in found and any(k in found for k in _QUALIFIED_KINDS):
        synth_tokens = len(re.findall(r"synth", lower))
        qual_tokens = len(
            re.findall(r"(?:tropical|future|futuristic|dubstep|edm|big(?:-| )?room|house|techno|trance|trap|dnb|jungle|hardstyle|phonk|synthwave|garage|amapiano|afro(?:-| )?house|grime)(?:\s+(?:edm|bass))?\s+synth", lower)
        )
        if synth_tokens <= qual_tokens:
            found.remove("synth")
    # "deep house" / "afro house" are one kind each, not house + deep_house
    if "house" in found and "deep_house" in found:
        found.remove("house")
    if "house" in found and "afro_house" in found:
        found.remove("house")
    return found


def _mix_stem_of(instrument: str) -> str:
    """Fold any instrument onto a Demucs mixer stem."""
    if instrument in MIX_STEMS:
        return instrument
    return "other"


def extract_mix_gains(prompt: str) -> dict[str, float]:
    """Parse per-stem level wishes like 'drums louder, vocals quieter' or
    'vocals -3dB drums +6dB' or 'prioritize drums over vocals'.

    Returns {mix_stem: gain_db} clamped to [-12, +12]. Empty if nothing found.
    """
    lower = prompt.lower()
    mentioned = extract_instruments(prompt)
    if not mentioned:
        return {}
    gains: dict[str, float] = {}

    # explicit per-instrument dB: "drums +6db", "vocals: -3 db", "bass at -4dB"
    # (trailing s? so "drums +6dB" matches kw "drum")
    for inst in mentioned:
        kws = _INSTRUMENT_KEYWORDS.get(inst, [inst])
        for kw in sorted(kws, key=len, reverse=True):
            m = re.search(rf"{re.escape(kw)}s?\s*(?:at|to|by)?\s*([+-]?\d+(?:\.\d+)?)\s*db\b", lower)
            if m:
                try:
                    gains[_mix_stem_of(inst)] = max(-12.0, min(12.0, float(m.group(1))))
                    break
                except ValueError:
                    pass
        else:
            continue
        continue
    if gains:
        return gains

    # "X louder ... Y quieter / lower / softer / down" in one prompt
    louder = {"louder", "up", "higher", "forward", "prominent", "punchier", "stronger"}
    quieter = {"quieter", "quiet", "down", "lower", "softer", "back", "buried", "muted", "reduce"}
    # split on "and", ",", "but", "while", "over", "vs"
    chunks = re.split(r"\s+(?:and|but|while|whereas)\s+|,|;|\s+over\s+|\s+vs\.?\s+", lower)
    for chunk in chunks:
        chunk_insts: list[str] = []
        for inst in mentioned:
            if any(kw in chunk for kw in _INSTRUMENT_KEYWORDS.get(inst, [])):
                chunk_insts.append(inst)
        if not chunk_insts:
            continue
        is_up = any(w in chunk for w in ("louder", "turn it up", "turn up", "raise", "increase", "boost", "bring up", "push up", "more "))
        is_up = is_up or ("up" in chunk and "turn" in chunk) or "louder" in chunk
        is_down = any(w in chunk for w in ("quieter", "quiet", "turn down", "lower", "reduce", "softer", "sit back", "push back", "less "))
        # bare direction words
        if not is_up and not is_down:
            if any(w in chunk for w in louder):
                is_up = True
            if any(w in chunk for w in quieter):
                is_down = True
        if is_up and not is_down:
            for inst in chunk_insts:
                gains[_mix_stem_of(inst)] = 6.0
        elif is_down and not is_up:
            for inst in chunk_insts:
                gains[_mix_stem_of(inst)] = -6.0
    if gains:
        return gains

    # "prioritize drums" / "feature the vocals" / "bring drums forward"
    # only the head (before over/vs/and) gets boosted; the tail gets dipped
    m = re.search(r"(?:priorit(?:ize|ise)|feature|spotlight|bring\s+\w+\s+forward|push\s+\w+\s+forward)\s+(?:the\s+)?([a-z &,\-]+)", lower)
    if m:
        tail = m.group(1)
        head = re.split(r"\s+over\s+|\s+vs\.?\s+|\s+and\s+|,", tail)[0]
        for inst in mentioned:
            if any(kw in head for kw in _INSTRUMENT_KEYWORDS.get(inst, [])):
                gains[_mix_stem_of(inst)] = 6.0
        # de-prioritize the rest mentioned alongside ("over vocals" tail)
        rest = [i for i in mentioned if _mix_stem_of(i) not in gains]
        for inst in rest:
            gains.setdefault(_mix_stem_of(inst), -3.0)
        if gains:
            return gains
    return gains


_FILLER_PATTERN = re.compile(r"(?<![a-z])(um+|uh+|ah+|er+)s?(?![a-z])")


STYLE_KEYWORDS = {
    # deep_house first: "deep house" contains the substring "house"
    "deep_house": ["deep house", "deep-house", "deep_house", "deep house style"],
    "house": ["house music", "house style", "four-on-the-floor", "four on the floor", "french house"],
    "techno": ["techno style", "melodic techno", "peak-time techno", "techno groove"],
    "trance": ["trance style", "uplifting trance", "psytrance", "psy-trance"],
    "trap": ["trap style", "trap edm", "hybrid trap"],
    "dnb": ["drum and bass style", "drum & bass", "dnb style", "jungle style"],
    "hardstyle": ["hardstyle", "hard style", "rawstyle"],
    "phonk": ["phonk style", "drift phonk"],
    "synthwave": ["synthwave", "retrowave", "outrun style", "synth-wave"],
    "garage": ["uk garage", "garage style", "2-step", "2step", "speed garage"],
    "amapiano": ["amapiano", "amapiano style", "log drum"],
    "afro_house": ["afro house", "afrohouse", "afro-house"],
    "jungle": ["jungle style", "amen break"],
    "grime": ["grime", "eskibeat", "eski beat"],
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
    # voice-audibility phrasing ("make her voice audible", "voice more clear")
    "make voice", "make her voice", "make his voice", "make the voice",
    "voice clear", "voices clear", "vocal clear", "speech clear",
    "voice audible", "vocal audible", "voices audible", "more audible",
    "cant hear the voice", "can't hear the voice", "cannot hear the voice",
    "hear her", "hear him", "hear the voice", "hear the vocal",
    "background noise", "background hiss", "background sound",
    "muffled", "muffle", "crisp vocal", "crisp voice",
    "clean up the voice", "clean the voice", "clean the vocal",
    "clean up vocals", "clean vocals", "clean up vocal",
    "clear up the voice", "clear up voice",
]

# Drum-piece keywords for selective adds ("add only snare", "kick drum").
_DRUM_PART_KEYWORDS: dict[str, list[str]] = {
    "kick": ["kick", "bass drum", "kick drum"],
    "snare": ["snare", "snare drum", "rim", "rimshot", "rim shot", "clap"],
    "hats": ["hat", "hi-hat", "hihat", "high hat", "cymbal", "ride", "crash",
             "shaker", "open hat", "closed hat"],
}


def voice_flags(prompt: str) -> dict:
    """Shared vocal-clarity params: aggression, audibility lift, mix-denoise."""
    lower = prompt.lower()
    voice_words = ("vocal", "voice", "voices", "speech", "speak", "sing",
                   "dialogue", "podcast", "singer", "vocalist", "narrat")
    noise_words = ("noise", "hiss", "disturbance", "denoise", "de-noise",
                   "hum", "buzz", "background")
    return {
        "aggressive": bool(re.search(
            r"\b(very|really|super|extremely|heavily|completely|totally|a lot|all( the)? (background )?noise|much noise)\b", lower)),
        "level_boost": bool(re.search(
            r"audible|can'?t hear|cannot hear|loud and clear|bring .*voice (up|forward|out)|voice.*(louder|up front)", lower)),
        "denoise_mix": (any(k in lower for k in noise_words)
                        and not any(k in lower for k in voice_words)),
    }


def extract_drum_parts(prompt: str) -> list[str] | None:
    """Which drum pieces to synthesize. None = full kit.

    - "add drums" (no specific piece named) -> None (full kit)
    - "add only snare" / "just the kick drum" -> ["snare"] / ["kick"]
    - "add kick and snare" -> ["kick", "snare"]
    - "add drums without hats" -> ["kick", "snare"]
    """
    lower = prompt.lower()
    if not re.search(r"\b(drum|drums|kick|snare|hat|hihat|hi-hat|cymbal|percussion|rim|rimshot|clap|ride|crash|shaker)\b", lower):
        return None
    found: list[str] = []
    for part, kws in _DRUM_PART_KEYWORDS.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}s?\b", lower):
                found.append(part)
                break
    if not found:
        return None  # generic "drums"/"percussion" -> full kit
    # exclusions: "drums without hats", "no cymbals", "except the kick"
    excluded = set()
    for part, kws in _DRUM_PART_KEYWORDS.items():
        for kw in kws:
            if re.search(rf"(?:without|except|minus|no)\s+(?:the\s+)?{re.escape(kw)}s?\b", lower):
                excluded.add(part)
    found = [p for p in found if p not in excluded]
    if not found:
        return None
    # "kick drum" alone (singular, no plural "drums"/"hats"/"snares") = kick only.
    # But "kick drums" / "drums" plural with no other piece = full kit.
    if len(found) == 1 and re.search(r"\bdrums\b", lower):
        others_named = any(
            kw in lower
            for part, kws in _DRUM_PART_KEYWORDS.items() if part != found[0]
            for kw in kws
        )
        if not others_named:
            return None
    return found


def extract_groove(prompt: str) -> str:
    lower = prompt.lower()
    if any(k in lower for k in ["hardstyle", "rawstyle", "reverse bass"]):
        return "hardstyle"
    if any(k in lower for k in ["uk garage", "speed garage", "2-step", "2step", "two-step", "two step"]) or re.search(r"\bgarage\b", lower):
        return "garage"
    if "amapiano" in lower or "log drum" in lower or "logdrum" in lower:
        return "amapiano"
    if any(k in lower for k in ["afro house", "afrohouse", "afro-house"]):
        return "afro_house"
    if any(k in lower for k in ["grime", "eskibeat", "eski"]):
        return "grime"
    if (any(k in lower for k in ["jungle", "amen break"])
            or re.search(r"\bamen\b", lower)):
        return "jungle"
    if any(k in lower for k in ["drum and bass", "drum & bass", "dnb"]):
        return "dnb"
    if any(k in lower for k in ["trap edm", "hybrid trap"]) or ("trap" in lower and any(w in lower for w in ("drum", "kick", "snare", "hat", "cymbal", "percussion", "808"))):
        return "trap"
    if any(k in lower for k in ["psytrance", "psy-trance", "uplifting trance", "trance"]):
        return "trance"
    if "techno" in lower:
        return "techno"
    if "deep house" in lower or "deep-house" in lower:
        return "deep_house"
    if "tech house" in lower or "french house" in lower:
        return "tech_house"
    if re.search(r"\bhouse\b", lower):
        return "house"
    if any(k in lower for k in ["synthwave", "retrowave", "outrun"]):
        return "synthwave"
    if "phonk" in lower:
        return "phonk"
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
    # stem mixer ("drums louder, vocals quieter", "balance the mix", "prioritize drums"):
    # needs 2+ stems, or 1 stem + an explicit mix/balance/prioritize word.
    # Runs before GAIN/BOOST so "vocals -3dB drums +6dB" mixes instead of mastering.
    _gains = extract_mix_gains(prompt)
    _mix_words = ("balance", "rebalance", "prioritiz", "prioritise", "feature the",
                  "bring forward", "push forward", "sit back", " fader", "levels",
                  " in the mix", "of the mix", "stem mix")
    if _gains and (len(_gains) >= 2 or any(w in lower for w in _mix_words) or "mix" in lower):
        return Intent.MIX
    # precise gain knob ("set gain +3dB", "volume to 80%") beats the fuzzy BOOST
    if re.search(r"[+-]?\d+(?:\.\d+)?\s*db\b", lower) or re.search(r"\bgain\b", lower):
        return Intent.GAIN
    if any(
        w in lower
        for w in ["set volume", "volume to ", "increase volume", "reduce volume", "lower the volume", "raise the volume", "turn the volume up", "turn the volume down", "make it quieter"]
    ):
        return Intent.GAIN
    style = extract_style(prompt)
    # \bstyle\b so "hardstyle" doesn't count as the word "style"
    if style and (any(w in lower for w in ["convert", "make it", "turn into", "remix"]) or re.search(r"\bstyle\b", lower)):
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
        # "add techno drums" = drums with a techno groove (single add), not a combine
        _groove_kinds = _GROOVE_KINDS
        _explicit_combine = any(w in lower for w in ["both", "together", "combine", "comebin", "merge", " and ", "layer them", "mix them"])
        _percussive = any(w in lower for w in ("drum", "kick", "snare", "hat", "cymbal", "percussion", "breakbeat", "break beat"))
        if len(mentioned) >= 2:
            if not (set(mentioned) <= ({"drums"} | _groove_kinds) and _percussive and not _explicit_combine):
                return Intent.COMBINE
            # else: fall through to single ADD (drums + groove kind)
        if extract_instrument(prompt):
            return Intent.ADD_INSTRUMENT
        if extract_wave(prompt):
            return Intent.ADD_INSTRUMENT
        if any(k in lower for k in ["bass", "drum", "synth", "guitar", "piano", "keys", "pad", "strings", "tropical", "future", "futuristic", "dubstep", "wobble", "edm", "big room", "bigroom", "lead", "pluck", "supersaw", "riddim", "808", "acid", "reese", "techno", "trance", "trap", "hardstyle", "phonk", "synthwave", "dnb", "house", "garage", "amapiano", "afro", "jungle", "grime", "brass", "sax", "trumpet", "flute", "violin", "cello", "harp", "marimba", "choir", "organ", "arp", "square"]):
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
    if any(w in lower for w in ["darker", "brighter", "energetic", "calm", "mood", "feel"]) or re.search(r"\btone\b", lower):
        return Intent.MOOD
    if any(w in lower for w in ["speed", "tempo", "ramp"]) or re.search(
        r"\b(fast|faster|slow|slower|speed ?up|speed ?down)\b", lower
    ):
        return Intent.SPEED
    if any(w in lower for w in ["reverb", "echo", "delay", "space"]):
        return Intent.REVERB
    if re.search(r"\breverse[sd]?\b", lower) or re.search(r"\bbackwards\b", lower):
        if "reverse bass" not in lower:
            return Intent.REVERSE
    if re.search(r"\b(loop|loops|looping|repeat|repeats)\b", lower):
        return Intent.REPEAT
    if (re.search(r"\btranspose\b", lower) or re.search(r"\bsemitones?\b", lower)
            or re.search(r"\bpitch\b", lower)
            or re.search(r"\btune\s+(it\s+)?(up|down|higher|lower)\b", lower)):
        return Intent.TRANSPOSE
    return Intent.UNKNOWN


def extract_format(prompt: str) -> str | None:
    formats = ["mp3", "wav", "flac", "aac", "ogg", "m4a"]
    lower = prompt.lower()
    for fmt in formats:
        # word boundary so "synthwave" doesn't match "wav"
        if re.search(rf"\b{re.escape(fmt)}\b", lower):
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
        params["fade_in"] = bool(
            re.search(r"fade[\s-]*in\b", lower) or ("fade" in lower and re.search(r"\bin\b", lower))
        )
        params["fade_out"] = bool(
            re.search(r"fade[\s-]*out\b", lower) or ("fade" in lower and re.search(r"\bout\b", lower))
        )
        # "fade in 2.5s and fade out 1s" / "3s fade out" — per-side durations
        m_in = re.search(r"fade[\s-]*in[^\d]*(\d+(?:\.\d+)?)\s*s", lower) or re.search(
            r"(\d+(?:\.\d+)?)\s*s\w*\s+fade[\s-]*in", lower
        )
        m_out = re.search(r"fade[\s-]*out[^\d]*(\d+(?:\.\d+)?)\s*s", lower) or re.search(
            r"(\d+(?:\.\d+)?)\s*s\w*\s+fade[\s-]*out", lower
        )
        if m_in:
            params["fade_in_dur"] = float(m_in.group(1))
        if m_out:
            params["fade_out_dur"] = float(m_out.group(1))

    if intent == Intent.GAIN:
        import math

        lower = prompt.lower()
        m_db = re.search(r"([+-]?\d+(?:\.\d+)?)\s*db\b", lower)
        m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", lower)
        if m_db:
            params["gain_db"] = max(-24.0, min(24.0, float(m_db.group(1))))
        elif m_pct:
            pct = max(1.0, float(m_pct.group(1)))
            params["gain_db"] = round(20 * math.log10(pct / 100.0), 1)
        elif any(w in lower for w in ["up", "raise", "increase", "louder"]):
            params["gain_db"] = 3.0
        elif any(w in lower for w in ["down", "lower", "reduce", "quieter"]):
            params["gain_db"] = -3.0
        else:
            params["gain_db"] = 0.0

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

    if intent == Intent.REVERSE:
        # region already in start/end when given; absent = whole track
        pass

    if intent == Intent.REPEAT:
        lower = prompt.lower()
        m = re.search(r"(\d+)\s*(?:x|times)\b", lower)
        if m:
            params["times"] = max(2, min(8, int(m.group(1))))
        else:
            params["times"] = 2

    if intent == Intent.TRANSPOSE:
        lower = prompt.lower()
        m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*semitones?\b", lower)
        if m:
            semi = float(m.group(1))
            # unsigned number ("down 3 semitones") takes direction from words
            if not m.group(1).startswith(("+", "-")):
                if re.search(r"\b(down|lower|drop|decrease)\b", lower):
                    semi = -abs(semi)
                elif re.search(r"\b(up|higher|raise|increase)\b", lower):
                    semi = abs(semi)
        else:
            m2 = re.search(r"(?:pitch|transpose|tune)(?:\s+it)?\s+(up|down|higher|lower)\b", lower)
            if m2:
                semi = 2.0 if m2.group(1) in ("up", "higher") else -2.0
            elif re.search(r"\b(higher|up)\b", lower):
                semi = 2.0
            elif re.search(r"\b(lower|down)\b", lower):
                semi = -2.0
            else:
                semi = 2.0
        params["semitones"] = max(-12.0, min(12.0, semi))

    if intent == Intent.ADD_INSTRUMENT:
        # ensure we have an instrument even if keyword was "synth" mapped to keys
        if not params.get("instrument"):
            lower = prompt.lower()
            wave = extract_wave(prompt)
            if wave == "808":
                params["instrument"] = "808"
            elif wave in ("supersaw",):
                params["instrument"] = "supersaw"
            elif wave in ("acid",):
                params["instrument"] = "acid"
            elif wave in ("reese",):
                params["instrument"] = "reese"
            elif wave in ("pluck",):
                params["instrument"] = "pluck"
            elif wave in ("pad",):
                params["instrument"] = "pad"
            elif wave in ("arp",):
                params["instrument"] = "arp"
            elif wave in ("square",):
                params["instrument"] = "square_lead"
            elif "hardstyle" in lower:
                params["instrument"] = "hardstyle"
            elif "phonk" in lower:
                params["instrument"] = "phonk"
            elif "synthwave" in lower or "retrowave" in lower or "outrun" in lower:
                params["instrument"] = "synthwave"
            elif "dnb" in lower or "drum and bass" in lower or "drum & bass" in lower:
                params["instrument"] = "dnb"
            elif "jungle" in lower or re.search(r"\bamen\b", lower):
                params["instrument"] = "jungle"
            elif "grime" in lower or "eski" in lower:
                params["instrument"] = "grime"
            elif "amapiano" in lower or "log drum" in lower or "logdrum" in lower:
                params["instrument"] = "amapiano"
            elif "afro house" in lower or "afrohouse" in lower or "afro-house" in lower:
                params["instrument"] = "afro_house"
            elif "garage" in lower or "2-step" in lower or "2step" in lower:
                params["instrument"] = "garage"
            elif re.search(r"\btrap\b", lower):
                params["instrument"] = "trap"
            elif "trance" in lower or "psytrance" in lower:
                params["instrument"] = "trance"
            elif "techno" in lower:
                params["instrument"] = "techno"
            elif "deep house" in lower:
                params["instrument"] = "deep_house"
            elif "house" in lower:
                params["instrument"] = "house"
            elif "dubstep" in lower or "wobble" in lower or "riddim" in lower:
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
        wave = extract_wave(prompt)
        if wave:
            params["wave"] = wave
        params["groove"] = extract_groove(prompt)
        m_bpm = re.search(r"(\d{2,3})\s*bpm", prompt.lower())
        if m_bpm:
            params["target_bpm"] = float(m_bpm.group(1))
        # selective drum pieces ("add only snare", "add a kick drum")
        if params.get("instrument") == "drums":
            parts = extract_drum_parts(prompt)
            if parts:
                params["drum_parts"] = parts
        # combined request from the other direction: clarify the voice first
        # (reachable via the LLM path; the regex classifier already prefers
        # ENHANCE_VOCALS when enhance keywords are present)
        if any(k in prompt.lower() for k in ENHANCE_KEYWORDS):
            params["also_enhance"] = True
            params.update(voice_flags(prompt))

    if intent == Intent.STYLE:
        style = extract_style(prompt)
        if style:
            params["style"] = style
        params["groove"] = extract_groove(prompt)

    if intent == Intent.ENHANCE_VOCALS:
        lower = prompt.lower()
        params["denoise"] = any(k in lower for k in ["noise", "hiss", "disturbance", "denoise", "clean", "audible", "clear", "hear", "crisp", "muffl"])
        params["clarity"] = True
        params.update(voice_flags(prompt))
        # Combined request ("add drums + make the voice clear"): voice first,
        # then layer the instrument quietly so the voice stays on top.
        # NB: voice words ("voice"/"vocal") name the enhance target, not the
        # thing to add — so skip them when picking the added instrument.
        if any(w in lower for w in ["add ", "generate", "create", "insert", "layer", "synthesize"]):
            mentioned = extract_instruments(prompt)
            add_inst = next((i for i in mentioned if i not in ("vocals", "choir")), None)
            if add_inst is None:
                single = extract_instrument(prompt)
                add_inst = single if single not in ("vocals", None) else None
            if add_inst and add_inst != "vocals":
                params["add_instrument"] = add_inst
                params["add_groove"] = extract_groove(prompt)
                if add_inst == "drums":
                    parts = extract_drum_parts(prompt)
                    if parts:
                        params["drum_parts"] = parts
                wave = extract_wave(prompt)
                if wave:
                    params["add_wave"] = wave
                m_bpm = re.search(r"(\d{2,3})\s*bpm", lower)
                if m_bpm:
                    params["add_target_bpm"] = float(m_bpm.group(1))

    if intent == Intent.COMBINE:
        params["instruments"] = extract_instruments(prompt) or ["drums", "bass"]
        # keep only generatable layers (drop vocals etc. that can't be synthesized)
        generatable = set(INSTRUMENT_CATALOG.keys()) - {"vocals"}
        params["instruments"] = [i for i in params["instruments"] if i in generatable] or ["drums", "bass"]
        wave = extract_wave(prompt)
        if wave:
            params["wave"] = wave
        params["groove"] = extract_groove(prompt)
        m_bpm = re.search(r"(\d{2,3})\s*bpm", prompt.lower())
        if m_bpm:
            params["target_bpm"] = float(m_bpm.group(1))
        # selective drum piece inside a combo ("add snare and bass")
        if params["instruments"] == ["drums"]:
            parts = extract_drum_parts(prompt)
            if parts:
                params["drum_parts"] = parts
        # voice-clarity half of the combo ("...and clean up the voice"):
        # clarify first, then layer (voice-first chaining in execute_plan)
        if any(k in prompt.lower() for k in ENHANCE_KEYWORDS):
            params["also_enhance"] = True
            params.update(voice_flags(prompt))
            params["denoise"] = True
            params["clarity"] = True

    if intent == Intent.MIX:
        gains = extract_mix_gains(prompt)
        # Mix Lab sliders send explicit JSON-ish "vocals -3dB drums +6dB" — trust it.
        # Fallback: prioritize the named instrument, dip the rest slightly.
        if not gains and instrument:
            gains = {_mix_stem_of(instrument): 6.0}
        params["gains_db"] = gains or {"drums": 6.0, "vocals": -3.0}
        params["groove"] = extract_groove(prompt)

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
