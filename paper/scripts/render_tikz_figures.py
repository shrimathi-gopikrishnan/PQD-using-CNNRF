"""Render the two TikZ flowchart figures used in main.tex as PNGs.

These mirror figures/architecture.tex and figures/deployment.tex so the
DOCX export can include a faithful, non-LaTeX-dependent rendering.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

ARROW = dict(arrowstyle="-|>", mutation_scale=14, lw=1.2, color="black")


def _box(ax, x, y, w, h, text, fill, fontsize=8.5, weight_first=True):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.0, edgecolor="black", facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)
    return (x, y, w, h)


def _arrow(ax, p1, p2, label=None, label_pos=0.5, label_offset=(0, 0.05)):
    arr = FancyArrowPatch(p1, p2, **ARROW)
    ax.add_patch(arr)
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * label_pos
        my = p1[1] + (p2[1] - p1[1]) * label_pos
        ax.text(
            mx + label_offset[0], my + label_offset[1], label,
            fontsize=7, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
        )


def render_architecture():
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")

    w, h = 1.85, 1.25
    y_top = 2.10
    xs = [1.05, 3.15, 5.25, 7.35, 9.45]

    in_text = "$\\bf{Input\\ window}$\n100 samples\n@ 5 kHz"
    s1_text = "$\\bf{Stage\\ 1}$\nThreshold rule\nRMS, THD$_{lo}$,\n$K, \\Delta_{max}, \\sigma_R$\n< 1 ms"
    s2_text = "$\\bf{Stage\\ 2}$\nHybrid CNN + RF\nraw → 1D-CNN\n→ 64-d → RF\n~67 ms"
    kb_text = "$\\bf{KB\\ enrichment}$\nseverity, cause,\nequipment, actions"
    push_text = "$\\bf{Dashboard}$\nSocketIO\npqd_result"

    in_box   = _box(ax, xs[0], y_top, w, h, in_text,   "#e0e0e0")
    s1_box   = _box(ax, xs[1], y_top, w, h, s1_text,   "#d6e4f5")
    s2_box   = _box(ax, xs[2], y_top, w, h, s2_text,   "#a8c5e8")
    kb_box   = _box(ax, xs[3], y_top, w, h, kb_text,   "#cdebc7")
    push_box = _box(ax, xs[4], y_top, w, h, push_text, "#f7d6a8")

    norm_text = "$\\bf{Pure\\_Sinusoidal}$\nfast-path output"
    norm_box = _box(ax, xs[2], 0.55, 1.7, 0.7, norm_text, "#f9e2c5", fontsize=8)

    def right_of(b): return (b[0] + b[2] / 2, b[1])
    def left_of(b):  return (b[0] - b[2] / 2, b[1])
    def bottom_of(b): return (b[0], b[1] - b[3] / 2)
    def top_of(b):   return (b[0], b[1] + b[3] / 2)

    _arrow(ax, right_of(in_box), left_of(s1_box))
    _arrow(ax, right_of(s1_box), left_of(s2_box), label="Abnormal", label_offset=(0, 0.18))
    _arrow(ax, right_of(s2_box), left_of(kb_box))
    _arrow(ax, right_of(kb_box), left_of(push_box))

    p1 = bottom_of(s1_box)
    elbow = (p1[0], 0.55)
    _arrow(ax, p1, elbow)
    _arrow(ax, elbow, left_of(norm_box), label="Normal", label_offset=(0, 0.18))

    p2 = right_of(norm_box)
    elbow2 = (push_box[0], p2[1])
    _arrow(ax, p2, elbow2)
    _arrow(ax, elbow2, bottom_of(push_box))

    out = FIG_DIR / "architecture.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def render_deployment():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")

    src_w, src_h = 1.30, 0.55
    src_x = 0.85
    grid = _box(ax, src_x, 2.55, src_w, src_h, "AC grid", "#fff4c2", fontsize=9)
    pv   = _box(ax, src_x, 1.85, src_w, src_h, "Solar PV", "#fff4c2", fontsize=9)
    dg   = _box(ax, src_x, 1.15, src_w, src_h, "DG / EV",  "#fff4c2", fontsize=9)

    proc_w, proc_h = 1.95, 1.05
    sense = _box(ax, 3.30, 1.85, proc_w, proc_h,
                 "$\\bf{Sensing}$\nZMPT101B (V)\nACS712 (I)\nADC @ 5 kHz", "#d6e4f5", fontsize=8.5)
    edge  = _box(ax, 5.65, 1.85, proc_w, proc_h,
                 "$\\bf{Edge\\ node}$\nRaspberry Pi /\nESP32 / ARM\nStage 1 + Stage 2", "#a8c5e8", fontsize=8.5)
    bus   = _box(ax, 8.00, 1.85, proc_w, proc_h,
                 "$\\bf{Comms}$\nRS485 / Modbus\nor SocketIO", "#cdebc7", fontsize=8.5)

    hmi_w, hmi_h = 1.95, 0.70
    scada = _box(ax, 10.05, 2.35, hmi_w, hmi_h, "$\\bf{SCADA\\ HMI}$", "#f7d6a8", fontsize=9)
    dash  = _box(ax, 10.05, 1.35, hmi_w, hmi_h, "$\\bf{Browser\\ dashboard}$", "#f7d6a8", fontsize=9)

    def right_of(b): return (b[0] + b[2] / 2, b[1])
    def left_of(b):  return (b[0] - b[2] / 2, b[1])

    sense_left = left_of(sense)
    _arrow(ax, right_of(grid), sense_left)
    _arrow(ax, right_of(pv),   sense_left)
    _arrow(ax, right_of(dg),   sense_left)
    _arrow(ax, right_of(sense), left_of(edge))
    _arrow(ax, right_of(edge),  left_of(bus))
    _arrow(ax, right_of(bus),   left_of(scada))
    _arrow(ax, right_of(bus),   left_of(dash))

    out = FIG_DIR / "deployment.png"
    fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render_architecture()
    render_deployment()
