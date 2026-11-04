"""Paraphrase expansion for the seed dataset.

Two conservative strategies keep every generated prompt grammatical and
label-preserving:

1. politeness wrappers ("please ...", "can you ... for me")
2. phrase substitutions ("remove" -> "delete", "keep only" -> "isolate", ...)

Generated rows inherit the seed row's labels and group. The group is what keeps
paraphrases on one side of the train/test split.
"""

from __future__ import annotations

import random
import re

from .schema import Label

# Ordered phrase substitutions. Each key is a regex; values are replacements.
# They are applied at most once per variant so labels cannot drift.
SUBSTITUTIONS: list[tuple[str, list[str]]] = [
    (r"\bremove the\b", ["delete the", "take out the", "get rid of the", "strip the"]),
    (r"\bget rid of the\b", ["remove the", "delete the", "take out the"]),
    (r"\btake out the\b", ["remove the", "delete the"]),
    (r"\bdelete the\b", ["remove the", "take out the"]),
    (r"\bkeep only\b", ["keep just", "leave only", "isolate"]),
    (r"\bkeep just\b", ["keep only", "leave only"]),
    (r"\bisolate the\b", ["extract the", "solo the", "pull out the"]),
    (r"\bcut from\b", ["remove from", "delete from"]),
    (r"\bcut out the\b", ["remove the", "delete the"]),
    (r"\bkeep only from\b", ["just keep from", "trim to"]),
    (r"\bnormalize\b", ["even out", "level out", "balance"]),
    (r"\beven out\b", ["normalize", "level out"]),
    (r"\bclean up\b", ["denoise", "remove the noise from"]),
    (r"\badd reverb to\b", ["put reverb on", "apply reverb to"]),
    (r"\bspeed it up\b", ["make it faster", "play it faster"]),
    (r"\bslow it down\b", ["make it slower", "play it slower"]),
    (r"\bmake it faster\b", ["speed it up", "play it faster"]),
    (r"\bmake it slower\b", ["slow it down", "play it slower"]),
    (r"\bmake the (\w+) louder\b", [r"turn up the \1", r"boost the \1", r"raise the \1"]),
    (r"\bmake the (\w+) quieter\b", [r"turn down the \1", r"lower the \1", r"reduce the \1"]),
    (r"\bturn up the (\w+)\b", [r"boost the \1", r"raise the \1"]),
    (r"\bturn down the (\w+)\b", [r"lower the \1", r"reduce the \1"]),
    (r"\bremove the silence\b", ["cut the silence", "strip the silence"]),
    (r"\bremove the pauses\b", ["cut the pauses", "delete the pauses"]),
    (r"\badd a fade out\b", ["put a fade out", "apply a fade out"]),
    (r"\badd a fade in\b", ["put a fade in", "apply a fade in"]),
    (r"\bbackground noise\b", ["background hiss", "noise in the background"]),
    (r"\bwhole track\b", ["whole song", "entire track"]),
    (r"\bwhole mix\b", ["entire mix", "full mix"]),
    (r"\bwhole song\b", ["whole track", "entire song"]),
]

PREFIXES = [
    "please ",
    "can you ",
    "i want to ",
    "i'd like to ",
    "could you ",
    "go ahead and ",
    "i need you to ",
]

SUFFIXES = [
    " please",
    " for me",
    " thanks",
    " when you can",
    " on this track",
]

SECOND_SUFFIXES = [
    " please",
    " for me",
    " thanks",
]

# Prefixes only work in front of a verb. "can you just the piano" is not
# English, so prompts that start with a noun phrase only get a suffix.
VERB_START = re.compile(
    r"^(please\s+)?(remove|delete|take|get|keep|isolate|extract|solo|fade|add|put|"
    r"apply|make|turn|boost|lower|raise|reduce|increase|normalize|even|level|clean|"
    r"denoise|cut|trim|crop|shorten|speed|slow|set|change|play|strip|mute|drop|kill|"
    r"pull|give|separate|balance|leave)\b"
)

# Amount formatting only; never changes the value.
AMOUNT_PATTERNS: list[tuple[str, str]] = [
    (r"\b(\d+(?:\.\d+)?) db\b", r"\1db"),
    (r"\b(\d+(?:\.\d+)?)db\b", r"\1 db"),
    (r"\b(\d+(?:\.\d+)?) seconds\b", r"\1s"),
    (r"\b(\d+(?:\.\d+)?)s\b", r"\1 seconds"),
]


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _apply_substitution(text: str, rng: random.Random) -> str | None:
    candidates = [(pattern, repls) for pattern, repls in SUBSTITUTIONS if re.search(pattern, text, re.I)]
    if not candidates:
        return None
    rng.shuffle(candidates)
    for pattern, repls in candidates:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        repl = rng.choice(repls)
        candidate = text[: match.start()] + match.expand(repl) + text[match.end() :]
        if candidate != text:
            return candidate
    return None


def _apply_politeness(text: str, rng: random.Random) -> str:
    base = text.rstrip(" .!?")
    prefix = ""
    if VERB_START.search(base) and not base.startswith("please") and rng.random() < 0.55:
        prefix = rng.choice(PREFIXES)
    suffix = ""
    if rng.random() < 0.45:
        candidate_suffix = rng.choice(SUFFIXES)
        tail = base.split()[-2:]
        # Do not repeat a politeness word that is already at the end.
        if not any(word in tail for word in candidate_suffix.split()):
            suffix = candidate_suffix
    return prefix + base + suffix


def _apply_amount_format(text: str, rng: random.Random) -> str:
    patterns = list(AMOUNT_PATTERNS)
    rng.shuffle(patterns)
    for pattern, repl in patterns:
        if re.search(pattern, text, re.I):
            return re.sub(pattern, repl, text, count=1, flags=re.I)
    return text


def variants_for(seed: Label, n_variants: int = 4, seed_value: int = 0) -> list[Label]:
    """Return up to `n_variants` label-preserving paraphrases of one seed row."""
    rng = random.Random(f"{seed.id}:{seed_value}")
    out: list[Label] = []
    seen = {normalize_text(seed.text)}

    attempts = 0
    while len(out) < n_variants and attempts < n_variants * 12:
        attempts += 1
        text = seed.text
        used_substitution = False

        if rng.random() < 0.75:
            replaced = _apply_substitution(text, rng)
            if replaced is not None:
                text = replaced
                used_substitution = True
        if not used_substitution or rng.random() < 0.5:
            text = _apply_politeness(text, rng)
        if rng.random() < 0.35:
            text = _apply_amount_format(text, rng)
        text = " ".join(text.split()).strip()

        norm = normalize_text(text)
        if norm in seen or len(norm.split()) < 2:
            continue
        seen.add(norm)
        out.append(
            Label(
                id=f"{seed.id}-v{len(out) + 1}",
                text=text,
                ops=list(seed.ops),
                targets=list(seed.targets),
                amount=seed.amount,
                unit=seed.unit,
                span=list(seed.span) if seed.span is not None else None,
                notes=seed.notes,
                source="template",
                group=seed.group or seed.id,
            )
        )
    return out


def expand(seeds: list[Label], variants: int = 4) -> list[Label]:
    rows: list[Label] = [
        Label(
            id=seed.id,
            text=seed.text,
            ops=list(seed.ops),
            targets=list(seed.targets),
            amount=seed.amount,
            unit=seed.unit,
            span=list(seed.span) if seed.span is not None else None,
            notes=seed.notes,
            source="seed",
            group=seed.group or seed.id,
        )
        for seed in seeds
    ]
    # Seeds win over generated variants; drop any paraphrase that duplicates
    # an existing prompt.
    seen = {normalize_text(row.text) for row in rows}
    for seed in seeds:
        for variant in variants_for(seed, variants):
            norm = normalize_text(variant.text)
            if norm in seen:
                continue
            seen.add(norm)
            rows.append(variant)
    return rows
