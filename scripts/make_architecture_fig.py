"""Generate Fig. 1/2 for the report: data flow + system architecture diagrams."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "report_figs" / "fig_architecture.png"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)

BOX = dict(boxstyle="round,pad=0.35", facecolor="#eef2f7", edgecolor="#1F3864", linewidth=1.2)
FONT = dict(fontsize=8.5, ha="center", va="center", weight="bold")


def draw_flow(ax, title, nodes, edges):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title(title, fontsize=9, weight="bold", pad=6)
    pos = {}
    for name, (x, y) in nodes.items():
        w = 2.6
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.45), w, 0.9, **BOX))
        ax.text(x, y, name, **FONT)
        pos[name] = (x, y)
    for a, b in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.annotate("", xy=(x2, y2 - 0.55), xytext=(x1, y1 + 0.55),
                    arrowprops=dict(arrowstyle="-|>", color="#1F3864", lw=1.2))


# Left: data flow
nodes_l = {
    "User": (1.6, 3.0),
    "UI (upload +\nprompt)": (4.0, 3.0),
    "Intent Parsing\n(LLM)": (6.4, 3.0),
    "Task\nDispatcher": (8.3, 3.0),
    "DSP Module": (7.3, 5.0),
    "AI Module": (9.2, 5.0),
    "Output\nGenerator": (8.3, 1.0),
    "Result": (5.3, 1.0),
}
edges_l = [
    ("User", "UI (upload +\nprompt)"),
    ("UI (upload +\nprompt)", "Intent Parsing\n(LLM)"),
    ("Intent Parsing\n(LLM)", "Task\nDispatcher"),
    ("Task\nDispatcher", "DSP Module"),
    ("Task\nDispatcher", "AI Module"),
    ("DSP Module", "Output\nGenerator"),
    ("AI Module", "Output\nGenerator"),
    ("Output\nGenerator", "Result"),
]
draw_flow(axes[0], "Fig. 1. Data flow of the proposed system", nodes_l, edges_l)


# Right: system architecture
nodes_r = {
    "User Interface\n(web + mobile)": (2.0, 5.2),
    "Intent Parsing\nLLM prompt engine": (5.2, 5.2),
    "Task Dispatcher\nintent -> module": (8.3, 5.2),
    "DSP Engine\nFFmpeg / Pydub / Librosa": (3.4, 2.6),
    "AI Engine\nDeepFilterNet, Demucs, Whisper": (7.2, 2.6),
    "Output Generator\nWAV / MP3 / stems ZIP": (5.2, 0.6),
}
edges_r = [
    ("User Interface\n(web + mobile)", "Intent Parsing\nLLM prompt engine"),
    ("Intent Parsing\nLLM prompt engine", "Task Dispatcher\nintent -> module"),
    ("Task Dispatcher\nintent -> module", "DSP Engine\nFFmpeg / Pydub / Librosa"),
    ("Task Dispatcher\nintent -> module", "AI Engine\nDeepFilterNet, Demucs, Whisper"),
    ("DSP Engine\nFFmpeg / Pydub / Librosa", "Output Generator\nWAV / MP3 / stems ZIP"),
    ("AI Engine\nDeepFilterNet, Demucs, Whisper", "Output Generator\nWAV / MP3 / stems ZIP"),
]
draw_flow(axes[1], "Fig. 2. System architecture (module-level)", nodes_r, edges_r)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
print("saved", OUT)