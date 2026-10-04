"""Phrase-level part generation — the fix for "static note" bass and pads.

The problem
-----------
`_bass_line` and friends played one figure on a loop, which is why an added
bass sounded like a held note. A part needs to *go somewhere*: new ideas every
couple of bars, approach tones into the next chord, rests, and a contour that
responds to the track's energy.

How this works
--------------
1. `scripts/train_phrasing_model.py` extracts real note sequences (pitch,
   onset, duration) from real music, using Demucs to isolate a monophonic
   stem and `librosa.pyin` to track it.
2. A small MLP learns `state -> (next interval, note length, velocity)`.
3. At generation time we walk that model one step at a time, constrained to the
   target track's key and instrument range, and render each note through the
   trained timbre model.

The learned part is rhythm and phrasing. The sound is ours (trained timbre +
our drum bank). Nothing from the training data ships.

STATUS: untested. Not yet run, trained, or verified.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "phrase_model.pt"

MAJOR = (0, 2, 4, 5, 7, 9, 11)
NAT_MINOR = (0, 2, 3, 5, 7, 8, 10)

STATE_DIM = 5
# Interval and note length are discrete musical choices, so they are predicted
# as distributions and *sampled*. An earlier version regressed one number for
# each and rounded it; that returns the conditional mean, which collapsed every
# line to near-zero intervals -- audibly a held note with a short run-up.
DELTA_BINS = 15          # -7 .. +7 semitones
DUR_BINS = 16            # 0 (rest) .. 15 sixteenth steps
VEL_DIM = 1
ACT_DIM = DELTA_BINS + DUR_BINS + VEL_DIM
PHRASE_STEPS = 32        # two bars of sixteenths
SAMPLE_TEMP = 1.0        # 1.0 = use the learned distribution as-is

_lock = threading.Lock()
_cache: dict | None = None


def load():
    """Load the phrase model. Raises if it has not been trained yet."""
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"phrase model not found at {MODEL_PATH}. Build it with:\n"
                "  python -m scripts.train_phrasing_model"
            )
        import torch

        blob = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        net = _build(blob["hidden"])
        net.load_state_dict(blob["state_dict"])
        net.eval()
        blob["net"] = net
        blob["torch"] = torch
        _cache = blob
        return _cache


def available() -> bool:
    return MODEL_PATH.exists()


def _build(hidden: int = 96):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(STATE_DIM, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, ACT_DIM),
            )

        def forward(self, x):
            return self.body(x)

    return Net()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - float(np.max(x))
    e = np.exp(x)
    s = float(e.sum())
    return (e / s) if s > 1e-12 else np.full_like(e, 1.0 / len(e))


def _snap_to_scale(midi: int, root: int, scale: tuple[int, ...]) -> int:
    """Nearest pitch that belongs to the scale, searching outward."""
    for d in range(0, 13):
        for cand in (midi - d, midi + d):
            if ((cand - root) % 12) in scale:
                return int(cand)
    return int(midi)


def generate_part(
    family: str,
    bpm: float,
    bars: int = 4,
    root_midi: int = 33,
    scale: tuple[int, ...] = NAT_MINOR,
    energy: list[float] | None = None,
    density: float = 1.0,
    sr: int = 22050,
    seed: int | None = None,
    pitch_lo: int = 28,
    pitch_hi: int = 96,
) -> tuple[np.ndarray, dict]:
    """Generate one musical part. Returns (mono audio, metadata).

    `energy` is an optional per-step intensity curve in 0..1 used to open the
    part up in loud sections and thin it out in quiet ones.
    """
    from server.ml.synthesis import timbre

    blob = load()
    torch = blob["torch"]
    net = blob["net"]

    rng = np.random.default_rng(
        seed if seed is not None else abs(hash((family, bpm, bars, root_midi))) % (2**31)
    )

    bpm = float(max(40.0, min(220.0, bpm)))
    step = 60.0 / bpm / 4.0
    total_steps = max(1, bars * 16)
    total_n = int(total_steps * step * sr) + sr

    buf = np.zeros(total_n, dtype=np.float32)
    notes: list[dict] = []
    rests = 0

    prev_midi = int(root_midi + 12)
    phrase_start_midi = prev_midi
    prev_loud = 0.0
    since_reset = 0
    t = 0
    guard = 0
    max_iter = total_steps * 4

    # ---- pass 1: plan the notes -------------------------------------------
    # Planning happens before rendering so that a run of same-pitch notes can be
    # merged into one sustained note. Re-striking the same pitch every step
    # sounds like a stutter, which is the same complaint as a static note.
    plan: list[dict] = []
    rests = 0

    while t < total_steps and guard < max_iter:
        guard += 1

        # A new phrase every two bars, so the model starts a fresh idea rather
        # than repeating one figure for the whole track.
        if since_reset >= PHRASE_STEPS:
            phrase_start_midi = prev_midi
            since_reset = 0

        e = 1.0
        if energy and len(energy) > t:
            e = float(energy[t])

        # State layout must match `scripts/train_phrasing_model.py` exactly.
        # Slot 3 is the TRACK's local loudness, not the note's own -- the model
        # was trained that way. Slot 4 is the last note's loudness.
        state = torch.tensor(
            [[
                (prev_midi - phrase_start_midi) / 24.0,
                (t % PHRASE_STEPS) / PHRASE_STEPS,
                (prev_midi - root_midi) / 24.0,
                e,
                prev_loud,
            ]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            out = net(state)[0].numpy()

        # Sample the discrete choices instead of taking their mean -- this is
        # what makes the line move.
        d = float(np.clip(density, 0.0, 1.0))
        d_p = _softmax(out[:DELTA_BINS] / SAMPLE_TEMP)
        u_p = _softmax(out[DELTA_BINS:DELTA_BINS + DUR_BINS] / SAMPLE_TEMP)
        vel = float(np.clip(out[-1], 0.0, 1.0))
        delta = int(rng.choice(DELTA_BINS, p=d_p)) - (DELTA_BINS // 2)

        # The model decides WHERE rests fall; the arrangement decides HOW MANY.
        # Left entirely to the model the part came out ~10% filled -- closer to
        # silence than to an instrument. Blending the model's rest-vs-note odds
        # with a target fill (geometric mean) keeps its local sense of space
        # while guaranteeing a usable density.
        target_fill = float(np.clip(0.30 + 0.50 * e * d, 0.15, 0.85))
        note_share = float(max(1.0 - u_p[0], 1e-6))
        model_odds = float(u_p[0]) / note_share
        target_odds = (1.0 - target_fill) / max(target_fill, 1e-6)
        blended = float(np.sqrt(max(model_odds, 1e-6) * target_odds))
        p_rest = blended / (1.0 + blended)

        if rng.random() < p_rest:
            dur_steps = 0.0
        else:
            tail = u_p[1:]
            ssum = float(tail.sum())
            dur_steps = float(
                rng.choice(np.arange(1, DUR_BINS), p=tail / ssum if ssum > 1e-9
                           else np.full(DUR_BINS - 1, 1.0 / (DUR_BINS - 1)))
            )

        if dur_steps < 1.0:
            rests += 1
            t += 1
            since_reset += 1
            continue

        dur_steps = int(min(dur_steps, total_steps - t))
        if dur_steps < 1:
            break
        midi = _snap_to_scale(
            int(np.clip(prev_midi + delta, pitch_lo, pitch_hi)), root_midi, scale
        )
        # Velocity also carries energy, but gently -- the arrangement does the
        # heavy lifting, not pushing every note harder.
        velocity = int(np.clip(40 + 80 * (0.45 * vel + 0.55 * e), 20, 127))

        if plan and plan[-1]["step"] + plan[-1]["steps"] == t and plan[-1]["midi"] == midi:
            # Same pitch, no gap: hold it instead of re-striking.
            plan[-1]["steps"] += dur_steps
            plan[-1]["velocity"] = max(plan[-1]["velocity"], velocity)
        else:
            plan.append({"step": t, "midi": midi, "steps": dur_steps,
                         "velocity": velocity})

        prev_midi = midi
        prev_loud = float(np.clip(velocity / 127.0, 0.0, 1.0))
        t += dur_steps
        since_reset += dur_steps

    # Never hand back silence. If the plan came out empty -- a very quiet track,
    # or a seed the model disliked -- fall back to a plain alternating root/fifth
    # figure. A feature that quietly adds nothing is worse than a plain one.
    if not plan and total_steps >= 4:
        degs = (0, 7, 0, 7)
        for k in range(0, total_steps, 2):
            d = degs[(k // 2) % len(degs)]
            plan.append({
                "step": k,
                "midi": _snap_to_scale(root_midi + d, root_midi, scale),
                "steps": 2,
                "velocity": 80,
            })

    # ---- pass 2: render the planned notes ---------------------------------
    notes: list[dict] = []
    for item in plan:
        note_len = item["steps"] * step * 0.94
        try:
            y, _ = timbre.render_note(
                family, item["midi"], duration=note_len,
                velocity=item["velocity"], sr=sr,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
        except Exception:
            continue
        s = int(item["step"] * step * sr)
        e2 = min(s + len(y), total_n)
        if s < e2 and np.any(y):
            buf[s:e2] += y[: e2 - s]
            notes.append(dict(item))

    peak = float(np.max(np.abs(buf))) + 1e-9
    out = (buf / peak * 0.85).astype(np.float32)

    # Contiguous runs of notes, so we can report how many separate ideas the
    # part actually had rather than just a note count.
    runs = 0
    prev_end = -99
    for n_ in notes:
        if n_["step"] != prev_end:
            runs += 1
        prev_end = n_["step"] + n_["steps"]

    meta = {
        "notes": len(notes),
        "rests": rests,
        "phrases": runs,
        "distinct_pitches": len({n_["midi"] for n_ in notes}),
        "phrasing_source": "trained on real note sequences",
        "bpm": bpm,
        "bars": bars,
        "seconds": round(len(out) / sr, 3),
        # Exposed so tests (and the UI) can verify the part is musical rather
        # than just "audio came out".
        "pitches": [n_["midi"] for n_ in notes],
        "note_steps": [n_["step"] for n_ in notes],
        "note_lengths": [n_["steps"] for n_ in notes],
    }
    return out, meta
