"""Generate Fig. 1/2 for the report: data flow + system architecture diagrams.

Clean vertical flowcharts with runtime validation:
  - no two boxes overlap
  - no arrow segment passes through an unrelated box
  - every label fits inside its box (box sizes come from measured text extents)
  - every box is inside the axis limits
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "report_figs" / "fig_architecture.png"

BOX_FC = "#eef2f7"
BOX_EC = "#1F3864"
ARROW_C = "#1F3864"
FONT_SIZE = 8.5


def box_rect(x, y, w, h, pad=0.12):
    return (x - w / 2 - pad, y - h / 2 - pad, w + 2 * pad, h + 2 * pad)


def intersects(a, b, margin=1e-3):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    return not (
        ax0 + aw <= bx0 + margin
        or bx0 + bw <= ax0 + margin
        or ay0 + ah <= by0 + margin
        or by0 + bh <= ay0 + margin
    )


def segment_hits_rect(p1, p2, rect, margin=1e-3):
    """True if the segment actually enters the rectangle (Liang-Barsky).

    A bounding-box test is too eager once boxes are wide: an arrow that only
    passes near a box would be reported as a collision.
    """
    x0, y0, w, h = rect
    x_min, x_max = x0 + margin, x0 + w - margin
    y_min, y_max = y0 + margin, y0 + h - margin
    if x_min > x_max or y_min > y_max:
        return False

    (ax_, ay_), (bx_, by_) = p1, p2
    dx, dy = bx_ - ax_, by_ - ay_
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, ax_ - x_min),
        (dx, x_max - ax_),
        (-dy, ay_ - y_min),
        (dy, y_max - ay_),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def validate(nodes, edges, widths, heights, xlim, ylim, label):
    rects = {n: box_rect(*nodes[n], widths[n], heights[n]) for n in nodes}
    names = list(nodes)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if intersects(rects[names[i]], rects[names[j]]):
                raise RuntimeError(
                    f"[{label}] boxes overlap: {names[i]!r} vs {names[j]!r}"
                )

    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        p1 = (xa, ya - heights[a] / 2)
        p2 = (xb, yb + heights[b] / 2)
        for n in names:
            if n in (a, b):
                continue
            if segment_hits_rect(p1, p2, rects[n]):
                raise RuntimeError(
                    f"[{label}] arrow {a!r}->{b!r} passes through box {n!r}"
                )

    for n in names:
        x, y = nodes[n]
        x0, y0, w, h = rects[n]
        if x0 < xlim[0] or x0 + w > xlim[1] or y0 < ylim[0] or y0 + h > ylim[1]:
            raise RuntimeError(f"[{label}] box {n!r} outside axis limits")


def draw_flow(ax, nodes, edges, xlim=(0, 6.0), ylim=(-1.2, 10)):
    """nodes: {name: (x, y)} centered; edges: [(from, to)]. Vertical flow."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_aspect("equal")

    # Measure the rendered text instead of estimating characters: the old
    # CHAR_W estimate was half the real width, so long labels hung outside
    # their boxes while the validator still reported "labels fit".
    texts = {
        name: ax.text(
            x, y, name, fontsize=FONT_SIZE, ha="center", va="center",
            weight="bold", zorder=4,
        )
        for name, (x, y) in nodes.items()
    }
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    widths, heights = {}, {}
    for name, text in texts.items():
        bb = text.get_window_extent(renderer)
        (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        # The patch's own 0.12 pad is added outside these dimensions, so only a
        # small breathing margin belongs here.
        widths[name] = (x1 - x0) + 0.10
        heights[name] = (y1 - y0) + 0.06

    for name, (x, y) in nodes.items():
        w, h = widths[name], heights[name]
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.12,rounding_size=0.1",
                facecolor=BOX_FC, edgecolor=BOX_EC, linewidth=1.2, zorder=2,
            )
        )

    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax.annotate(
            "",
            xy=(xb, yb + heights[b] / 2),
            xytext=(xa, ya - heights[a] / 2),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.3, mutation_scale=14),
            zorder=1,
        )

    validate(nodes, edges, widths, heights, xlim, ylim, ax.get_title())
    return widths, heights


fig, axes = plt.subplots(1, 2, figsize=(12, 7.2), dpi=150)

# ---------------- LEFT: data flow (vertical flowchart) ----------------
nodes_l = {
    "User": (3.0, 9.2),
    "UI (upload + prompt)": (3.0, 7.5),
    "Intent Parsing\n(grammar + optional LLM)": (3.0, 5.8),
    "Task Dispatcher": (3.0, 4.1),
    "DSP Module": (1.85, 2.5),
    "AI Module": (4.15, 2.5),
    "Output Generator": (3.0, 1.3),
    "Result": (3.0, -0.2),
}
edges_l = [
    ("User", "UI (upload + prompt)"),
    ("UI (upload + prompt)", "Intent Parsing\n(grammar + optional LLM)"),
    ("Intent Parsing\n(grammar + optional LLM)", "Task Dispatcher"),
    ("Task Dispatcher", "DSP Module"),
    ("Task Dispatcher", "AI Module"),
    ("DSP Module", "Output Generator"),
    ("AI Module", "Output Generator"),
    ("Output Generator", "Result"),
]
axes[0].set_title("Fig. 1. Data flow of the proposed system", fontsize=9, weight="bold", pad=8)
draw_flow(axes[0], nodes_l, edges_l)

# ---------------- RIGHT: system architecture ----------------
nodes_r = {
    "User Interface\n(web app)": (3.0, 9.2),
    "Intent Parsing\nrule grammar (+ optional LLM)": (3.0, 7.5),
    "Task Dispatcher\n(intent to module)": (3.0, 5.8),
    "DSP Engine\nFFmpeg, pydub,\nlibrosa": (1.5, 3.6),
    "AI Engine\nDemucs, Whisper,\nRF/MLP models": (4.5, 3.6),
    "Output Generator\n(WAV / MP3 / ZIP)": (3.0, 1.7),
    "Storage + Download": (3.0, 0.4),
}
edges_r = [
    ("User Interface\n(web app)", "Intent Parsing\nrule grammar (+ optional LLM)"),
    ("Intent Parsing\nrule grammar (+ optional LLM)", "Task Dispatcher\n(intent to module)"),
    ("Task Dispatcher\n(intent to module)", "DSP Engine\nFFmpeg, pydub,\nlibrosa"),
    ("Task Dispatcher\n(intent to module)", "AI Engine\nDemucs, Whisper,\nRF/MLP models"),
    ("DSP Engine\nFFmpeg, pydub,\nlibrosa", "Output Generator\n(WAV / MP3 / ZIP)"),
    ("AI Engine\nDemucs, Whisper,\nRF/MLP models", "Output Generator\n(WAV / MP3 / ZIP)"),
    ("Output Generator\n(WAV / MP3 / ZIP)", "Storage + Download"),
]
axes[1].set_title("Fig. 2. System architecture (module-level)", fontsize=9, weight="bold", pad=8)
draw_flow(axes[1], nodes_r, edges_r)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
print("validation passed: no overlaps, no crossing arrows, labels fit")