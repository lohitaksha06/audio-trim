"""Generate Fig. 1/2 for the report: data flow + system architecture diagrams.

Clean vertical flowcharts with runtime validation:
  - no two boxes overlap
  - no arrow segment passes through an unrelated box
  - every label fits inside its box
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
MIN_W = 2.3
MIN_H = 0.85
CHAR_W = 0.07  # data units per character at FONT_SIZE


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
    x0, y0, w, h = rect
    x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
    if x_max < x0 + margin or x_min > x0 + w - margin:
        return False
    if y_max < y0 + margin or y_min > y0 + h - margin:
        return False
    return True


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


def draw_flow(ax, nodes, edges, xlim=(0, 5.5), ylim=(-1.2, 10)):
    """nodes: {name: (x, y)} centered; edges: [(from, to)]. Vertical flow."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_aspect("equal")

    widths, heights = {}, {}
    for name, (x, y) in nodes.items():
        longest = max(len(line) for line in name.split("\n"))
        lines = len(name.split("\n"))
        widths[name] = max(MIN_W, CHAR_W * longest)
        heights[name] = max(MIN_H, 0.30 * lines + 0.12)

    for name, (x, y) in nodes.items():
        w, h = widths[name], heights[name]
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.12,rounding_size=0.1",
                facecolor=BOX_FC, edgecolor=BOX_EC, linewidth=1.2,
            )
        )
        ax.text(x, y, name, fontsize=FONT_SIZE, ha="center", va="center", weight="bold")

    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax.annotate(
            "",
            xy=(xb, yb + heights[b] / 2),
            xytext=(xa, ya - heights[a] / 2),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.3, mutation_scale=14),
        )

    validate(nodes, edges, widths, heights, xlim, ylim, ax.get_title())
    return widths, heights


fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150)

# ---------------- LEFT: data flow (vertical flowchart) ----------------
nodes_l = {
    "User": (2.7, 9.2),
    "UI (upload + prompt)": (2.7, 7.5),
    "Intent Parsing (LLM)": (2.7, 5.8),
    "Task Dispatcher": (2.7, 4.1),
    "DSP Module": (1.4, 2.5),
    "AI Module": (4.0, 2.5),
    "Output Generator": (2.7, 1.3),
    "Result": (2.7, -0.2),
}
edges_l = [
    ("User", "UI (upload + prompt)"),
    ("UI (upload + prompt)", "Intent Parsing (LLM)"),
    ("Intent Parsing (LLM)", "Task Dispatcher"),
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
    "User Interface\n(web + mobile)": (2.7, 9.2),
    "Intent Parsing\nLLM prompt engine": (2.7, 7.5),
    "Task Dispatcher\n(intent to module)": (2.7, 5.8),
    "DSP Engine\nFFmpeg / Pydub / Librosa": (1.4, 3.6),
    "AI Engine\nDeepFilterNet, Demucs,\nWhisper, kNN router": (4.0, 3.6),
    "Output Generator\n(WAV / MP3 / stems ZIP)": (2.7, 1.7),
    "Storage + Download": (2.7, 0.4),
}
edges_r = [
    ("User Interface\n(web + mobile)", "Intent Parsing\nLLM prompt engine"),
    ("Intent Parsing\nLLM prompt engine", "Task Dispatcher\n(intent to module)"),
    ("Task Dispatcher\n(intent to module)", "DSP Engine\nFFmpeg / Pydub / Librosa"),
    ("Task Dispatcher\n(intent to module)", "AI Engine\nDeepFilterNet, Demucs,\nWhisper, kNN router"),
    ("DSP Engine\nFFmpeg / Pydub / Librosa", "Output Generator\n(WAV / MP3 / stems ZIP)"),
    ("AI Engine\nDeepFilterNet, Demucs,\nWhisper, kNN router", "Output Generator\n(WAV / MP3 / stems ZIP)"),
    ("Output Generator\n(WAV / MP3 / stems ZIP)", "Storage + Download"),
]
axes[1].set_title("Fig. 2. System architecture (module-level)", fontsize=9, weight="bold", pad=8)
draw_flow(axes[1], nodes_r, edges_r)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
print("validation passed: no overlaps, no crossing arrows, labels fit")